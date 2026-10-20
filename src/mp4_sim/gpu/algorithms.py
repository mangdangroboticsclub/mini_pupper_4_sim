"""PPO anchored to a frozen policy: rsl_rl 5.0.1's PPO plus a loss-side MSE between the actor's mean and the frozen
network's mean, evaluated on the clean ``anchor`` observation group, with a weight (beta) held, then decayed to a floor.
The anchor's input width is read from its weights; relative anchor paths resolve against the repository root;
``actor_warmup_iters`` (default 0) trains only the value function for the first updates, so the actor's first steps
follow a fitted critic.

``AnchorPPO.update`` is derived from rsl_rl's ``PPO.update`` (rsl_rl, BSD-3-Clause).
"""


from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO

from .. import paths


def _build_anchor_mlp(state_dict: dict, device: str) -> nn.Sequential:
    """Rebuild the frozen actor's mean net (n_in->128->128->64->12, ELU) and load it.

    ``actor_state_dict`` has keys ``mlp.{0,2,4,6}.{weight,bias}`` plus ``distribution.std_param``; only the mlp
    weights are loaded, strictly.
    """
    # nn.Linear construction draws from the global torch RNG; building the
    # anchor inside fork_rng keeps the training RNG stream identical to a
    # run without an anchor, so training with beta = 0 matches plain PPO exactly.
    with torch.random.fork_rng(devices=[]):
        n_in = state_dict["mlp.0.weight"].shape[1]
        net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 12),
        )
    mlp_keys = {k.removeprefix("mlp."): v for k, v in state_dict.items() if k.startswith("mlp.")}
    net.load_state_dict(mlp_keys, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def resolve_anchor_path(anchor_path: str) -> Path:
    """Relative anchor paths are resolved against the repository root, not against the trainer's working directory."""
    path = Path(anchor_path).expanduser()
    return path if path.is_absolute() else paths.REPO / path


class AnchorPPO(PPO):
    """PPO with an action-mean MSE anchor toward a frozen policy."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        anchor_path: str,
        anchor_sha256: str,
        anchor_beta0: float,
        anchor_hold_iters: int,
        anchor_decay_iters: int,
        anchor_beta_floor: float = 0.0,
        anchor_obs_group: str = "anchor",
        actor_warmup_iters: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("AnchorPPO does not support RND/symmetry (untested interaction).")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("AnchorPPO supports the MLP (non-recurrent) lane only.")
        if anchor_beta0 < 0.0:
            raise ValueError(f"anchor_beta0={anchor_beta0} < 0")
        path = resolve_anchor_path(anchor_path)
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        expected = str(anchor_sha256).strip().lower()
        if digest != expected:
            raise ValueError(
                f"Anchor checkpoint hash mismatch: {path} is {digest[:12]}..., "
                f"cfg registers {expected[:12]}..."
            )
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        self.anchor_mlp = _build_anchor_mlp(loaded["actor_state_dict"], self.device)
        self.anchor_obs_group = anchor_obs_group
        self.anchor_beta0 = float(anchor_beta0)
        self.anchor_hold_iters = int(anchor_hold_iters)
        self.anchor_decay_iters = int(anchor_decay_iters)
        self.anchor_beta_floor = float(anchor_beta_floor)
        self._anchor_update_count = 0
        self.actor_warmup_iters = int(actor_warmup_iters)

    # -- schedule ---------------------------------------------------------
    def anchor_beta_at(self, it: int) -> float:
        """beta at update-counter ``it`` (counting from launch)."""
        if it < self.anchor_hold_iters:
            return self.anchor_beta0
        if it < self.anchor_hold_iters + self.anchor_decay_iters:
            frac = (it - self.anchor_hold_iters) / max(self.anchor_decay_iters, 1)
            return max(self.anchor_beta_floor, self.anchor_beta0 * (1.0 - frac))
        return self.anchor_beta_floor

    # -- update: the stock rsl_rl loop plus one guarded insertion -----------
    def update(self) -> dict[str, float]:
        beta = self.anchor_beta_at(self._anchor_update_count)
        warmup = self._anchor_update_count < self.actor_warmup_iters
        self._anchor_update_count += 1

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_anchor_loss = 0.0
        mean_anchor_mse = 0.0
        mean_actor_grad_norm = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if warmup:          # value function only: the actor receives no gradient
                loss = self.value_loss_coef * value_loss

            # --- the anchor (skipped exactly at beta == 0, so that path is the
            # stock computation) --------------------------------------------
            if beta != 0.0 and not warmup:
                if self.anchor_obs_group not in batch.observations.keys():
                    raise KeyError(
                        f"AnchorPPO: obs group '{self.anchor_obs_group}' missing from rollout "
                        f"storage (available: {list(batch.observations.keys())}). The environment "
                        "must declare the clean anchor group."
                    )
                anchor_obs = batch.observations[self.anchor_obs_group]
                with torch.no_grad():
                    anchor_mu = self.anchor_mlp(anchor_obs)
                anchor_mse = (self.actor.output_mean - anchor_mu).pow(2).mean()
                loss = loss + beta * anchor_mse
                mean_anchor_mse += anchor_mse.item()
                mean_anchor_loss += beta * anchor_mse.item()

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_grad_norm += float(actor_grad_norm)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_anchor_loss /= num_updates
        mean_anchor_mse /= num_updates
        mean_actor_grad_norm /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "anchor": mean_anchor_loss,
            "anchor_mse": mean_anchor_mse,
            "anchor_beta": beta,
            "actor_grad_norm": mean_actor_grad_norm,
        }
        return loss_dict

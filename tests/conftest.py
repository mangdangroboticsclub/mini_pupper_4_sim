import pytest


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Tests marked ``gpu`` are skipped without CUDA."""
    if _cuda():
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs an NVIDIA GPU (CUDA)")

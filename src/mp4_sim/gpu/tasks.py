"""mjlab task registration.

Loaded by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml, so
mjlab's ``train`` and ``play`` commands can find this project's tasks. Must
stay importable without side effects beyond registration.
"""

"""Operator-algebra training environments built on top of Helix APIs."""

from typing import Any


def register_helix_envs(register: Any) -> None:
    from .registry import register_helix_envs as _register

    _register(register)


def load_environment(*args: Any, **kwargs: Any):
    """Alias for the default AF partition Verifiers environment loader."""

    from .af_partition import load_environment as _load

    return _load(*args, **kwargs)


def load_verifiers_environment(*args: Any, **kwargs: Any):
    from .af_partition.env import load_verifiers_environment as _load

    return _load(*args, **kwargs)


__all__ = ["register_helix_envs", "load_environment", "load_verifiers_environment"]

"""Registration helpers for Verifiers / Prime runtimes."""

from __future__ import annotations

from typing import Any


def register_helix_envs(register: Any) -> None:
    """Register environments with a Verifiers-compatible registry.

    Parameters
    ----------
    register:
        Callable ``register(env_id: str, loader: Callable[..., Any])`` similar to
        :func:`verifiers.registry.register_env`. The function intentionally accepts a
        generic callable so we can use it with different runtime hubs.
    """
    from .af_partition import load_verifiers_environment

    register("helix/af_partition:v0", load_verifiers_environment)


__all__ = ["register_helix_envs"]

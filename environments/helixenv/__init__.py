"""Operator-algebra training environments built on top of Helix APIs."""

from typing import Any, Optional


def register_helix_envs(register: Any) -> None:
    from .registry import register_helix_envs as _register

    _register(register)


def load_environment(
    *,
    system_prompt: Optional[str] = None,
    use_think: bool = False,
    seed: int = 1234,
    max_episodes: Optional[int] = None,
    enable_llm_judge: bool = False,
    llm_judge_model: str = "gpt-4.1-mini",
    llm_judge_base_url: str = "https://api.openai.com/v1",
    llm_judge_api_key_var: str = "OPENAI_API_KEY",
    llm_judge_prompt: Optional[str] = None,
    **env_kwargs: Any,
):
    """Alias for the default AF partition Verifiers environment loader."""

    from .af_partition import load_environment as _load

    return _load(
        system_prompt=system_prompt,
        use_think=use_think,
        seed=seed,
        max_episodes=max_episodes,
        enable_llm_judge=enable_llm_judge,
        llm_judge_model=llm_judge_model,
        llm_judge_base_url=llm_judge_base_url,
        llm_judge_api_key_var=llm_judge_api_key_var,
        llm_judge_prompt=llm_judge_prompt,
        **env_kwargs,
    )


def load_verifiers_environment(*args: Any, **kwargs: Any):
    from .af_partition.env import load_verifiers_environment as _load

    return _load(*args, **kwargs)


__all__ = ["register_helix_envs", "load_environment", "load_verifiers_environment"]

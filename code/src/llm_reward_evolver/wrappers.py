from __future__ import annotations

from typing import Any, Callable, Optional

from .reward import RewardProgram


class CustomRewardWrapper:
    """Gymnasium wrapper that replaces env reward with an LLM reward program."""

    def __init__(
        self,
        env: Any,
        reward_program: RewardProgram,
        progress_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CustomRewardWrapper requires gymnasium.") from exc

        class _Wrapper(gym.Wrapper):
            def __init__(self, wrapped_env: Any) -> None:
                super().__init__(wrapped_env)
                self._previous_obs = None

            def reset(self, **kwargs: Any):
                obs, info = self.env.reset(**kwargs)
                self._previous_obs = obs
                return obs, info

            def step(self, action: Any):
                next_obs, original_reward, terminated, truncated, info = self.env.step(action)
                progress = progress_provider() if progress_provider else 0.0
                obs = self._previous_obs if self._previous_obs is not None else next_obs
                custom_reward = reward_program(obs, action, next_obs, original_reward, info, progress)
                self._previous_obs = next_obs
                info = dict(info)
                info["original_reward"] = float(original_reward)
                info["custom_reward"] = float(custom_reward)
                return next_obs, custom_reward, terminated, truncated, info

        self.env = _Wrapper(env)

    def unwrap(self) -> Any:
        return self.env


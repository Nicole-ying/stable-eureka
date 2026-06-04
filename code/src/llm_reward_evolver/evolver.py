from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig
from .feedback import build_feedback, build_scalar_feedback
from .llm import LLMClient, build_llm_client, extract_code
from .prompts import build_initial_prompt, build_refine_prompt
from .reward import RewardProgram
from .trainer import inspect_env, make_env, train_agent


@dataclass
class IterationRecord:
    iteration: int
    score: float
    success_rate: float
    mean_episode_length: float
    converged: bool
    failure_mode: str
    reward_code_path: str
    interrupted: bool = False
    error_message: str = ""
    reward_error_count: int = 0
    reward_last_error: str = ""


class RewardEvolver:
    def __init__(self, config: ExperimentConfig, llm_client: Optional[LLMClient] = None) -> None:
        self.config = config
        self.llm_client = llm_client or build_llm_client(config.llm_provider, config.llm_model)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[IterationRecord]:
        observation_desc, action_desc = inspect_env(self.config.env_name, self.config.seed)
        records: List[IterationRecord] = []
        current_code: Optional[str] = None
        best_code: Optional[str] = None
        best_score = float("-inf")
        feedback = ""

        for iteration in range(self.config.max_iterations):
            if current_code is None:
                prompt = build_initial_prompt(
                    self.config.env_name,
                    self.config.task_description,
                    observation_desc,
                    action_desc,
                    self.config.reward_structure,
                )
            else:
                prompt = build_refine_prompt(
                    self.config.env_name,
                    self.config.task_description,
                    current_code,
                    feedback,
                    best_code,
                    self.config.reward_structure,
                )

            current_code = self._generate_valid_reward(prompt, best_code)
            reward_path = self._write_text(f"reward_iter_{iteration}.py", current_code)
            reward_program = RewardProgram(
                current_code,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )

            result = train_agent(
                self.config.env_name,
                reward_program,
                self.config.total_timesteps,
                self.config.eval_episodes,
                self.config.target_score,
                self.config.seed + iteration,
                self.config.training_algorithm,
            )

            stats = result.stats
            if stats.mean_eval_score > best_score:
                best_score = stats.mean_eval_score
                best_code = current_code
                self._write_text("reward_best.py", current_code)

            record = IterationRecord(
                iteration=iteration,
                score=stats.mean_eval_score,
                success_rate=stats.success_rate,
                mean_episode_length=stats.mean_episode_length,
                converged=stats.converged,
                failure_mode=stats.failure_mode,
                reward_code_path=str(reward_path),
                interrupted=stats.interrupted,
                error_message=stats.error_message,
                reward_error_count=stats.reward_error_count,
                reward_last_error=stats.reward_last_error,
            )
            records.append(record)
            self._write_history(records)

            feedback = self._build_feedback(stats)
            if best_code and best_score > stats.mean_eval_score:
                feedback += (
                    "\nConservative rollback instruction:\n"
                    "- The latest reward is worse than the previous best reward. Start from previous_best_code, "
                    "not from the latest code.\n"
                    "- Keep original_reward as the dominant anchor and make only one small targeted change.\n"
                    "- Do not introduce a new large bonus, a new hard clamp, or a new global multiplier.\n"
                )
            self._write_text(f"feedback_iter_{iteration}.txt", feedback)

            stop, _reason = self._should_stop(records)
            if stop:
                break

        return records

    def _build_feedback(self, stats) -> str:
        if self.config.feedback_mode == "scalar":
            return build_scalar_feedback(stats)
        feedback = build_feedback(stats)
        if self.config.env_name.lower().startswith("mountaincar"):
            feedback += (
                "\nEnvironment-specific diagnostic for MountainCar:\n"
                "- The original reward is sparse (-1 per step), so dense shaping is needed.\n"
                "- Encourage progress toward the right goal position, but also reward absolute velocity "
                "because the car must first swing left and right to build momentum.\n"
                "- Avoid rewarding only rightward movement; that can prevent the agent from learning the "
                "backward swing needed for energy buildup.\n"
                "- A good HRDC reward should combine distance-to-goal improvement, velocity magnitude, "
                "goal bonus, and a mild step penalty, with early weights favoring momentum and later "
                "weights favoring goal approach.\n"
            )
        if self.config.env_name.lower().startswith("acrobot"):
            feedback += (
                "\nEnvironment-specific diagnostic for Acrobot:\n"
                "- The task is to swing the two-link arm upward, so dense shaping should reward "
                "tip height / upright posture rather than only survival.\n"
                "- Encourage useful angular velocity early for swing-up, then shift weights toward "
                "stability and high tip position later.\n"
                "- Penalize excessive action magnitude mildly, but avoid suppressing momentum too early.\n"
                "- Use angular velocity magnitude abs(thetaDot1)+abs(thetaDot2), not signed "
                "thetaDot1+thetaDot2.\n"
                "- Reward current tip height and target-height bonus, not only one-step height progress; "
                "one-step height progress alone is often too sparse.\n"
                "- Keep original_reward active at all training stages; do not multiply it by "
                "(1 - training_progress).\n"
                "- Do not manually clamp final reward inside compute_reward. The wrapper already "
                "clips reward, and a hard [-1, 1] clamp can erase height/goal differences.\n"
                "- For discrete action, use a scalar cost such as 0.05 if action != 1 else 0.0; "
                "do not use vector norms for action.\n"
                "- A good HRDC reward should combine height progress, angular momentum, stability near "
                "upright, and a small control cost.\n"
            )
        if self.config.env_name.lower().startswith("lunarlander"):
            feedback += (
                "\nEnvironment-specific diagnostic for LunarLander:\n"
                "- Reward approaching the landing pad center: penalize abs(x_position) and abs(y_position).\n"
                "- Reward slow, stable descent: penalize abs(x_velocity), abs(y_velocity), abs(angle), and "
                "abs(angular_velocity).\n"
                "- Add bonuses for left_leg_contact and right_leg_contact; both legs touching is a strong "
                "soft-landing signal.\n"
                "- Penalize fuel usage/actions mildly, especially main engine action 2, but avoid preventing "
                "necessary braking when downward velocity is high.\n"
                "- Preserve original_reward as a bounded anchor so the learned policy is still optimized "
                "for the environment's true landing score.\n"
                "- If the reward contains positive distance/velocity/angle penalties, flip their sign; "
                "bad distance, speed, tilt, angular speed, and fuel use must reduce reward.\n"
                "- Do not multiply the entire reward by (1 - training_progress); late training still "
                "needs strong contact and soft-landing rewards.\n"
                "- Do not use info['fuel_used'] or similar unavailable keys; infer fuel cost from action.\n"
                "- A good HRDC schedule should focus early on stabilizing velocity and angle, then later "
                "increase pad-centering and landing-contact rewards.\n"
                "- Treat action as a scalar integer in {0, 1, 2, 3}; do not use vector norms.\n"
            )
        return feedback

    def _generate_valid_reward(self, prompt: str, fallback_code: Optional[str]) -> str:
        last_error = ""
        for attempt in range(self.config.reward_repair_attempts + 1):
            response = self.llm_client.complete(prompt if attempt == 0 else self._repair_prompt(prompt, last_error))
            code = extract_code(response)
            try:
                RewardProgram(
                    code,
                    reward_clip=self.config.reward_clip,
                    error_fallback=self.config.reward_error_fallback,
                )
                self._smoke_test_reward_code(code)
                return code
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if fallback_code:
            return fallback_code
        return default_reward_code()

    def _repair_prompt(self, original_prompt: str, error: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous reward code failed validation with this error:\n"
            f"{error}\n"
            "Return a corrected compute_reward function only."
        )

    def _smoke_test_reward_code(self, code: str) -> None:
        env = make_env(self.config.env_name, self.config.seed)
        try:
            obs, _info = env.reset(seed=self.config.seed)
            action = env.action_space.sample()
            next_obs, original_reward, _terminated, _truncated, info = env.step(action)
            program = RewardProgram(
                code,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )
            program(obs, action, next_obs, float(original_reward), info, 0.25)
            if program.error_count:
                raise ValueError(f"Reward runtime smoke test failed: {program.last_error}")
            self._semantic_smoke_test(program)
        finally:
            env.close()

    def _semantic_smoke_test(self, program: RewardProgram) -> None:
        if not self.config.env_name.lower().startswith("lunarlander"):
            if self.config.env_name.lower().startswith("acrobot"):
                self._acrobot_semantic_smoke_test(program)
            return
        good_state = [0.0, 0.02, 0.0, -0.02, 0.0, 0.0, 1.0, 1.0]
        bad_state = [0.9, 1.1, 0.8, -1.0, 0.9, 1.5, 0.0, 0.0]
        near_no_contact = [0.0, 0.04, 0.0, -0.03, 0.0, 0.0, 0.0, 0.0]
        good_reward = program(good_state, 0, good_state, 0.0, {}, 0.8)
        bad_reward = program(bad_state, 2, bad_state, 0.0, {}, 0.8)
        contact_reward = program(near_no_contact, 0, good_state, 0.0, {}, 0.8)
        no_contact_reward = program(near_no_contact, 0, near_no_contact, 0.0, {}, 0.8)
        if good_reward <= bad_reward:
            raise ValueError(
                "LunarLander semantic smoke test failed: good landing state is not rewarded above bad state"
            )
        if contact_reward <= no_contact_reward:
            raise ValueError(
                "LunarLander semantic smoke test failed: two-leg contact does not improve reward"
            )

    def _acrobot_semantic_smoke_test(self, program: RewardProgram) -> None:
        low_state = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        high_state = [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        progress_state = [0.0, 1.0, 0.0, 1.0, 1.5, -1.5]
        low_reward = program(low_state, 1, low_state, -1.0, {}, 0.8)
        high_reward = program(low_state, 1, high_state, -1.0, {}, 0.8)
        progress_reward = program(low_state, 0, progress_state, -1.0, {}, 0.2)
        if high_reward <= low_reward:
            raise ValueError(
                "Acrobot semantic smoke test failed: high tip state is not rewarded above low state"
            )
        if progress_reward <= low_reward:
            raise ValueError(
                "Acrobot semantic smoke test failed: swing-up progress is not rewarded"
            )

    def _should_stop(self, records: List[IterationRecord]) -> tuple[bool, Optional[str]]:
        latest = records[-1]
        if latest.score >= self.config.target_score:
            return True, "target score reached"
        if len(records) >= self.config.max_iterations:
            return True, "max iterations reached"
        if len(records) <= self.config.force_iterations_before_patience:
            return False, None
        if len(records) >= self.config.patience:
            recent = records[-self.config.patience :]
            improvements = [
                recent[i + 1].score - recent[i].score for i in range(len(recent) - 1)
            ]
            if all(value < self.config.min_improvement for value in improvements):
                return True, "patience reached"
        return False, None

    def _write_text(self, name: str, text: str) -> Path:
        path = self.output_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def _write_history(self, records: List[IterationRecord]) -> None:
        path = self.output_dir / "history.json"
        data = [asdict(record) for record in records]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def default_reward_code() -> str:
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return float(original_reward)
"""

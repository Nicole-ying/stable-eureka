import re
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict, Any
import json

from stable_baselines3.common.vec_env import VecFrameStack, VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env, make_atari_env


def read_from_file(path: Path) -> str:
    with open(path, "r") as file:
        return file.read()


def get_code_from_response(response: str, regex: List[str]) -> str:
    for reg in regex:
        code = re.search(reg, response, re.DOTALL)
        if code:
            return code.group(1).strip()

    return ''


def append_and_save_to_txt(path: Path, txt: str):
    with open(path, 'a') as file:
        file.write(txt)


def save_to_txt(path: Path, txt: str):
    with open(path, 'w') as file:
        file.write(txt)


def save_to_json(path: Path, data: dict):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def read_from_json(path: Path) -> dict:
    with open(path, 'r') as file:
        return json.load(file)


def indent_code(code: str, signature: Optional[str] = None) -> str:
    indented_code = ''
    if signature:
        indented_code += f'    {signature}\n'
    indented_code += '\n'.join(['    ' + line for line in code.split('\n')])
    return indented_code


def make_env(env_class, env_kwargs, n_envs, is_atari, state_stack, multithreaded: bool = False) -> VecEnv:
    vec_env_cls = SubprocVecEnv if multithreaded and n_envs > 1 else DummyVecEnv

    if is_atari:
        env = make_atari_env(env_id=env_class, env_kwargs=env_kwargs, n_envs=n_envs, vec_env_cls=vec_env_cls)
    else:
        env = make_vec_env(env_id=env_class, env_kwargs=env_kwargs, n_envs=n_envs, vec_env_cls=vec_env_cls)

    if state_stack > 1:
        env = VecFrameStack(env, n_stack=state_stack)

    return env


_GLOBAL_POLICY_KEYS = {
    'fitness_score',
    'reward',
    'episode_length',
}

_BEHAVIOR_KEY_HINTS = (
    'terminal',
    'success_like',
    'safe_landing',
    'crash',
    'timeout',
    'out_of_bounds',
    'unsafe',
)


def _as_numeric_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if arr.size == 0:
        return None
    return arr


def _format_float(value: float) -> str:
    if value is None or not np.isfinite(value):
        return 'nan'
    if abs(value) >= 100:
        return f'{value:.1f}'
    return f'{value:.3f}'


def _series_stats(value: Any) -> Optional[Dict[str, float]]:
    arr = _as_numeric_array(value)
    if arr is None:
        return None

    return {
        'final': float(arr[-1]),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
        'nonzero_eval_fraction': float(np.mean(np.abs(arr) > 1e-9)),
    }


def _metric_group(key: str) -> str:
    if key in _GLOBAL_POLICY_KEYS:
        return 'global'
    if any(hint in key for hint in _BEHAVIOR_KEY_HINTS):
        return 'behavior'
    return 'component'


def _render_stats_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f'## {title}\nNone.\n'

    lines = [
        f'## {title}',
        '| name | final | mean | max | min | nonzero eval fraction | expert note |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        stats = row['stats']
        final = stats['final']
        nonzero = stats['nonzero_eval_fraction']
        if abs(final) < 1e-9 and nonzero == 0:
            note = 'dead or inactive in all recorded eval points; verify trigger condition and timing'
        elif abs(final) > 0 and abs(final) >= 5 * max(1.0, abs(stats['mean'])):
            note = 'large final value; check whether it dominates local behavior'
        elif abs(final) >= 25:
            note = 'large magnitude; compare against positive/negative signals and payment timing'
        else:
            note = 'inspect with reward code expression and intended behavior'

        lines.append(
            '| {name} | {final} | {mean} | {maxv} | {minv} | {nonzero:.2f} | {note} |'.format(
                name=row['name'],
                final=_format_float(stats['final']),
                mean=_format_float(stats['mean']),
                maxv=_format_float(stats['max']),
                minv=_format_float(stats['min']),
                nonzero=stats['nonzero_eval_fraction'],
                note=note,
            )
        )
    return '\n'.join(lines) + '\n'


def reflection_component_to_str(component: Dict):
    """Build a compact, expert-oriented reflection input from evals.json.

    The original implementation dumped raw Max/Mean/Min lines. That made the LLM
    see numbers but not the human-expert structure needed for reward debugging.
    This function deliberately separates global behavior, terminal behavior, and
    reward components, and it asks the LLM to reconstruct the *complete* reward
    component manifest from the current reward code.
    """
    grouped = {'global': [], 'behavior': [], 'component': []}
    for key, value in component.items():
        stats = _series_stats(value)
        if stats is None:
            continue
        grouped[_metric_group(key)].append({'name': key, 'stats': stats})

    for group in grouped.values():
        group.sort(key=lambda row: abs(row['stats']['final']), reverse=True)

    text = []
    text.append('# Compact training feedback for expert reward reflection')
    text.append('')
    text.append(_render_stats_table('Global policy metrics', grouped['global']))
    text.append(_render_stats_table('Behavior / terminal metrics', grouped['behavior']))
    text.append(_render_stats_table('Reward component actual returns', grouped['component']))
    text.append('## Mandatory expert interpretation task')
    text.append('Do not summarize the reward function by only listing dominant components.')
    text.append('For every component in the current reward code, extract the exact expression, condition, coefficient, payment timing, intended behavior, actual value from the table above, and diagnosis.')
    text.append('Payment timing must be one of: per-step, per-step cumulative, progress-delta, one-shot, terminal, terminal-gated, or unknown.')
    text.append('A component with actual value near zero must be diagnosed: unreachable condition, behavior never reached, wrong timing, or intentionally inactive.')
    text.append('A component with large magnitude must be diagnosed by payment timing: per-step accumulation and terminal penalties are not interchangeable.')
    return '\n'.join(text) + '\n'


def reward_history_to_str(history: List[Dict[str, Any]], max_full_codes: int = 3) -> str:
    """Render reward history without hiding component-level code.

    Every reward function is saved completely in reward_history.json. The prompt
    gets a compact index plus full code for the most recent records to avoid
    flooding the LLM while preserving exact historical designs.
    """
    if not history:
        return 'No previous reward history.'

    lines = [
        '# Reward function history index',
        'The complete reward code for every listed record is saved in code/reward_history/reward_history.json.',
        'Each reward function must be treated as a complete component manifest, not as a short main-design summary.',
        '',
        '| iter | sample | parent | status | fitness | reward code path |',
        '|---:|---:|---|---|---:|---|',
    ]

    for record in history:
        parent = record.get('parent', 'None')
        status = 'ELITE' if record.get('is_elite') else 'candidate'
        lines.append(
            '| {iter} | {sample} | {parent} | {status} | {fitness} | {path} |'.format(
                iter=record.get('iteration'),
                sample=record.get('sample'),
                parent=parent,
                status=status,
                fitness=_format_float(float(record.get('fitness', 0.0))),
                path=record.get('reward_code_path', ''),
            )
        )

    lines.append('')
    lines.append(f'# Full reward code for the most recent {min(max_full_codes, len(history))} history records')
    for record in history[-max_full_codes:]:
        lines.append('')
        lines.append('## iter_{:03d} sample_{} status={}'.format(
            int(record.get('iteration', -1)),
            record.get('sample'),
            'ELITE' if record.get('is_elite') else 'candidate',
        ))
        lines.append('```python')
        lines.append(record.get('reward_code', '').strip())
        lines.append('```')

    return '\n'.join(lines) + '\n'


def summarize_final_eval(evals: Dict[str, Any]) -> Dict[str, float]:
    summary = {}
    for key, value in evals.items():
        stats = _series_stats(value)
        if stats is not None:
            summary[key] = stats['final']
    return summary

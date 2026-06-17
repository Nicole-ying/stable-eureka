"""Compact prompt compiler for expert reward reflection.

Raw evidence may be saved on disk for human inspection, but the LLM should only
receive a small, task-relevant compact prompt. The goal is to prevent prompt
inflation across long 30+ iteration searches while preserving the information a
human reward-design expert actually needs.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional


def _clip(text: str, max_chars: int, label: str) -> str:
    text = (text or '').strip()
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED {label}: kept first {max_chars} chars. Raw file is saved on disk but not sent to LLM.]"
    )


def normalize_reward_code(code: str) -> str:
    """Normalize reward code for exact duplicate detection.

    This is deliberately conservative: it catches identical or formatting-only
    duplicates without pretending to solve semantic equivalence.
    """
    code = code or ''
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def reward_code_hash(code: str) -> str:
    return hashlib.sha256(normalize_reward_code(code).encode('utf-8')).hexdigest()[:16]


def compact_task_card(task_description: str, env_code: str = '', max_chars: int = 900) -> str:
    """Return a short task card. Do not send full env code after bootstrap."""
    task = (task_description or '').strip()
    lines = [line.strip() for line in task.splitlines() if line.strip()]
    summary = ' '.join(lines[:6]) if lines else 'Task description unavailable.'

    # Keep only one hint about the environment interface. Full env_code is noisy
    # and should not be repeatedly sent to the LLM in long searches.
    interface_hint = ''
    if env_code:
        compute_match = re.search(r'def\s+compute_reward\s*\(([^)]*)\)', env_code)
        step_match = re.search(r'def\s+step\s*\(([^)]*)\)', env_code)
        if compute_match:
            interface_hint = f"\nReward interface hint: compute_reward({compute_match.group(1).strip()})"
        elif step_match:
            interface_hint = f"\nEnvironment step interface hint: step({step_match.group(1).strip()})"

    return _clip(summary + interface_hint, max_chars, 'task card')


def _final_metric(summary: Dict[str, Any], key: str, default: str = 'NA') -> str:
    value = summary.get(key, default)
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def build_elite_chain_summary(history: List[Dict[str, Any]], max_records: int = 6) -> str:
    elites = [record for record in history if record.get('is_elite')]
    if not elites:
        return 'No elite has been confirmed yet.'

    elites = elites[-max_records:]
    lines = [
        '| elite | parent | fitness | success_like | episode_length | code_hash |',
        '|---|---|---:|---:|---:|---|',
    ]
    for record in elites:
        summary = record.get('final_eval_summary', {}) or {}
        label = f"iter_{int(record.get('iteration', -1)):03d}_sample_{record.get('sample')}"
        lines.append(
            '| {label} | {parent} | {fitness} | {success} | {ep_len} | {code_hash} |'.format(
                label=label,
                parent=record.get('parent') or 'None',
                fitness=f"{float(record.get('fitness', 0.0)):.3f}",
                success=_final_metric(summary, 'success_like_rate'),
                ep_len=_final_metric(summary, 'episode_length'),
                code_hash=record.get('code_hash', 'NA'),
            )
        )
    return '\n'.join(lines)


def build_failed_children_summary(
    history: List[Dict[str, Any]],
    parent_label: Optional[str],
    max_records: int = 8,
) -> str:
    if not parent_label:
        return 'No parent-specific child search history yet.'

    children = [record for record in history if record.get('parent') == parent_label]
    if not children:
        return f'No recorded children from {parent_label} yet.'

    children = children[-max_records:]
    lines = [
        f'Children searched from current parent {parent_label}:',
        '| child | status | fitness | success_like | episode_length | code_hash | avoid note |',
        '|---|---|---:|---:|---:|---|---|',
    ]
    for record in children:
        summary = record.get('final_eval_summary', {}) or {}
        label = f"iter_{int(record.get('iteration', -1)):03d}_sample_{record.get('sample')}"
        status = 'elite' if record.get('is_elite') else 'failed_or_non_elite_child'
        lines.append(
            '| {label} | {status} | {fitness} | {success} | {ep_len} | {code_hash} | do not repeat this exact code hash; diagnose changed components before reusing the same family |'.format(
                label=label,
                status=status,
                fitness=f"{float(record.get('fitness', 0.0)):.3f}",
                success=_final_metric(summary, 'success_like_rate'),
                ep_len=_final_metric(summary, 'episode_length'),
                code_hash=record.get('code_hash', 'NA'),
            )
        )
    return '\n'.join(lines)


def build_duplicate_guard(history: List[Dict[str, Any]], parent_label: Optional[str]) -> str:
    hashes = []
    for record in history:
        code_hash = record.get('code_hash')
        if not code_hash:
            continue
        label = f"iter_{int(record.get('iteration', -1)):03d}_sample_{record.get('sample')}"
        relation = 'child_of_current_parent' if parent_label and record.get('parent') == parent_label else 'history'
        hashes.append(f'- {code_hash}: {label}, {relation}, fitness={float(record.get("fitness", 0.0)):.3f}')

    if not hashes:
        return 'No previous reward hashes yet.'

    return 'Do not regenerate rewards with these exact normalized code hashes:\n' + '\n'.join(hashes[-20:])


def build_compact_reflection_prompt(
    *,
    reflection_init: str,
    reflection_end: str,
    task_description: str,
    env_code: str,
    component_feedback: str,
    parent_reward_code: str,
    parent_label: str,
    reward_history: List[Dict[str, Any]],
    max_total_chars: int = 15000,
) -> str:
    """Build a bounded expert prompt.

    The prompt intentionally excludes raw expert_memory_context, raw iteration
    evidence, full evals, full history code, previous LLM responses, and full env
    code. Those artifacts may remain on disk, but they must not enter the LLM
    prompt.
    """
    parent_reward_code = _clip(parent_reward_code, 5200, 'current parent reward code')
    component_feedback = _clip(component_feedback, 3600, 'component feedback')

    blocks = [
        '# Expert reward reflection input',
        'Use only the compact evidence below. Raw evidence exists on disk but is intentionally not provided to avoid noise and prompt inflation.',
        '',
        '## Task card',
        compact_task_card(task_description, env_code),
        '',
        '## Elite chain summary',
        _clip(build_elite_chain_summary(reward_history), 1800, 'elite chain summary'),
        '',
        '## Failed / non-elite children from current parent',
        _clip(build_failed_children_summary(reward_history, parent_label), 2200, 'children from current parent'),
        '',
        '## Duplicate guard',
        _clip(build_duplicate_guard(reward_history, parent_label), 1600, 'duplicate guard'),
        '',
        '## Current iteration compact component feedback',
        component_feedback,
        '',
        '## Current parent reward code to modify',
        f'Parent: {parent_label}',
        '```python',
        parent_reward_code,
        '```',
        '',
        '## Reflection rules',
        reflection_init.strip(),
        reflection_end.strip(),
        '',
        '## Output contract',
        'First output the complete component diagnosis table. Then output exactly one Python code block containing the revised compute_reward function. Do not output or request extra files.',
    ]

    prompt = '\n'.join(blocks)
    return _clip(prompt, max_total_chars, 'entire compact reflection prompt') + '\n'

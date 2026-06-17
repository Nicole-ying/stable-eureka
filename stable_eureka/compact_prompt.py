"""Build relevant reflection prompts without mechanical truncation.

Raw evidence is excluded from the LLM input, but exact reward information is not
truncated: elite reward code, current parent code, and component formulas must be
kept complete.
"""

import ast
import hashlib
import re
import textwrap
from typing import Any, Dict, List, Optional


def normalize_reward_code(code: str) -> str:
    code = re.sub(r'#.*', '', code or '')
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def reward_code_hash(code: str) -> str:
    return hashlib.sha256(normalize_reward_code(code).encode('utf-8')).hexdigest()[:16]


def _label(record: Dict[str, Any]) -> str:
    return f"iter_{int(record.get('iteration', -1)):03d}_sample_{record.get('sample')}"


def _metric(summary: Dict[str, Any], key: str) -> str:
    value = (summary or {}).get(key, 'NA')
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def task_card(task_description: str, env_code: str) -> str:
    lines = [line.strip() for line in (task_description or '').splitlines() if line.strip()]
    text = ' '.join(lines[:6]) if lines else 'Task description unavailable.'
    match = re.search(r'def\s+compute_reward\s*\(([^)]*)\)', env_code or '')
    if match:
        text += f"\nReward interface hint: compute_reward({match.group(1).strip()})"
    return text


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return '<unparse_failed>'


def component_manifest(code: str) -> str:
    """Return every component key and its formula/expression from reward code."""
    try:
        tree = ast.parse(textwrap.dedent(code or ''))
    except SyntaxError:
        return 'Could not parse code; use the full reward code as source of truth.'

    assignments: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            expr = _unparse(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = expr

    rows = ['| component | returned expression | formula |', '|---|---|---|']
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        ret = node.value
        comp_dict = ret.elts[1] if isinstance(ret, ast.Tuple) and len(ret.elts) >= 2 else ret
        if not isinstance(comp_dict, ast.Dict):
            continue
        for key_node, value_node in zip(comp_dict.keys, comp_dict.values):
            if key_node is None:
                continue
            key = str(key_node.value) if isinstance(key_node, ast.Constant) else _unparse(key_node)
            value = _unparse(value_node)
            formula = assignments.get(value, value)
            rows.append(f'| {key} | `{value}` | `{formula}` |')
            found = True
        break

    if found:
        return '\n'.join(rows)

    for name, expr in assignments.items():
        if any(word in name for word in ['reward', 'bonus', 'penalty', 'cost']):
            rows.append(f'| {name} | `{name}` | `{expr}` |')
            found = True
    return '\n'.join(rows) if found else 'No component dictionary found; use full reward code as source of truth.'


def elite_chain(history: List[Dict[str, Any]]) -> str:
    elites = [r for r in history if r.get('is_elite')]
    if not elites:
        return 'No elite has been confirmed yet.'
    rows = ['| elite | parent | fitness | success_like | episode_length | code_hash |', '|---|---|---:|---:|---:|---|']
    for r in elites:
        s = r.get('final_eval_summary', {})
        rows.append(f"| {_label(r)} | {r.get('parent') or 'None'} | {float(r.get('fitness', 0.0)):.3f} | {_metric(s, 'success_like_rate')} | {_metric(s, 'episode_length')} | {r.get('code_hash', 'NA')} |")
    return '\n'.join(rows)


def elite_reward_codes(history: List[Dict[str, Any]]) -> str:
    elites = [r for r in history if r.get('is_elite')]
    if not elites:
        return 'No elite reward code has been confirmed yet.'
    blocks = []
    for r in elites:
        code = r.get('reward_code', '')
        blocks += [
            f"## {_label(r)} full elite reward code",
            f"fitness={float(r.get('fitness', 0.0)):.3f}, parent={r.get('parent') or 'None'}, code_hash={r.get('code_hash', 'NA')}",
            'Component manifest:',
            component_manifest(code),
            '```python',
            code.strip(),
            '```',
        ]
    return '\n\n'.join(blocks)


def children_from_parent(history: List[Dict[str, Any]], parent_label: Optional[str]) -> str:
    if not parent_label:
        return 'No parent-specific child search history yet.'
    children = [r for r in history if r.get('parent') == parent_label]
    if not children:
        return f'No recorded children from {parent_label} yet.'
    blocks = [f'Children searched from current parent {parent_label}:']
    for r in children:
        s = r.get('final_eval_summary', {})
        status = 'elite' if r.get('is_elite') else 'failed_or_non_elite_child'
        code = r.get('reward_code', '')
        blocks += [
            f"## {_label(r)} | {status} | fitness={float(r.get('fitness', 0.0)):.3f} | success_like={_metric(s, 'success_like_rate')} | episode_length={_metric(s, 'episode_length')} | hash={r.get('code_hash', 'NA')}",
            'Component manifest:',
            component_manifest(code),
        ]
    return '\n'.join(blocks)


def duplicate_guard(history: List[Dict[str, Any]], parent_label: Optional[str]) -> str:
    lines = []
    for r in history:
        relation = 'child_of_current_parent' if parent_label and r.get('parent') == parent_label else 'history'
        lines.append(f"- {r.get('code_hash', 'NA')}: {_label(r)}, {relation}, fitness={float(r.get('fitness', 0.0)):.3f}")
    return 'Do not regenerate rewards with these exact normalized code hashes:\n' + '\n'.join(lines) if lines else 'No previous reward hashes yet.'


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
    max_total_chars: Optional[int] = None,
) -> str:
    # max_total_chars is ignored intentionally: clipping would remove component formulas or reward code.
    blocks = [
        '# Expert reward reflection input',
        'Only relevant evidence is included. Raw memory/evidence files and previous LLM responses are not included.',
        '## Task card', task_card(task_description, env_code),
        '## Elite chain summary', elite_chain(reward_history),
        '## Full elite reward functions and component formulas', elite_reward_codes(reward_history),
        '## Failed / non-elite children from current parent', children_from_parent(reward_history, parent_label),
        '## Duplicate guard', duplicate_guard(reward_history, parent_label),
        '## Current iteration compact component feedback', component_feedback.strip(),
        '## Current parent reward code to modify', f'Parent: {parent_label}',
        'Current parent component manifest:', component_manifest(parent_reward_code),
        '```python', parent_reward_code.strip(), '```',
        '## Reflection rules', reflection_init.strip(), reflection_end.strip(),
        '## Output contract',
        'First output the complete component diagnosis table. Then output exactly one Python code block containing the revised compute_reward function. Do not output or request extra files.',
    ]
    return '\n\n'.join(blocks) + '\n'

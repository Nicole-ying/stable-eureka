"""Compact prompt compiler for expert reward reflection.

This module does not mechanically truncate prompts. Its job is to remove noisy raw
artifacts from the LLM input while preserving the information that must be exact:
complete elite reward code, the current parent reward code, and every reward
component with its formula/expression.
"""

import ast
import hashlib
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple


def normalize_reward_code(code: str) -> str:
    """Normalize reward code for exact duplicate detection."""
    code = code or ''
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def reward_code_hash(code: str) -> str:
    return hashlib.sha256(normalize_reward_code(code).encode('utf-8')).hexdigest()[:16]


def compact_task_card(task_description: str, env_code: str = '') -> str:
    """Return a concise task card without sending full env code after bootstrap."""
    task = (task_description or '').strip()
    lines = [line.strip() for line in task.splitlines() if line.strip()]
    summary = ' '.join(lines[:6]) if lines else 'Task description unavailable.'

    interface_hint = ''
    if env_code:
        compute_match = re.search(r'def\s+compute_reward\s*\(([^)]*)\)', env_code)
        step_match = re.search(r'def\s+step\s*\(([^)]*)\)', env_code)
        if compute_match:
            interface_hint = f"\nReward interface hint: compute_reward({compute_match.group(1).strip()})"
        elif step_match:
            interface_hint = f"\nEnvironment step interface hint: step({step_match.group(1).strip()})"

    return summary + interface_hint


def _final_metric(summary: Dict[str, Any], key: str, default: str = 'NA') -> str:
    value = summary.get(key, default)
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def _label(record: Dict[str, Any]) -> str:
    return f"iter_{int(record.get('iteration', -1)):03d}_sample_{record.get('sample')}"


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return '<unparse_failed>'


def _extract_assignments_with_conditions(code: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract variable assignments and the condition path where they occur.

    This is not a validator. It is a lightweight manifest builder to help the
    reflection prompt show every reward component formula instead of vague
    "main design" summaries.
    """
    assignments: Dict[str, List[Tuple[str, str]]] = {}

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return assignments

    def visit_statements(statements: List[ast.stmt], conditions: List[str]):
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                expr = _safe_unparse(stmt.value)
                condition = ' and '.join(conditions) if conditions else 'always / current execution path'
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append((condition, expr))
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                expr = f"{stmt.target.id} {type(stmt.op).__name__}= {_safe_unparse(stmt.value)}"
                condition = ' and '.join(conditions) if conditions else 'always / current execution path'
                assignments.setdefault(stmt.target.id, []).append((condition, expr))
            elif isinstance(stmt, ast.If):
                test = _safe_unparse(stmt.test)
                visit_statements(stmt.body, conditions + [test])
                visit_statements(stmt.orelse, conditions + [f'not ({test})'])
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_statements(stmt.body, conditions)
            elif isinstance(stmt, (ast.For, ast.While, ast.With, ast.Try)):
                for field in ('body', 'orelse', 'finalbody'):
                    body = getattr(stmt, field, None)
                    if isinstance(body, list):
                        visit_statements(body, conditions)

    visit_statements(tree.body, [])
    return assignments


def _find_return_component_dict(code: str) -> Dict[str, str]:
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.Tuple) and len(value.elts) >= 2:
            maybe_dict = value.elts[1]
        else:
            maybe_dict = value
        if isinstance(maybe_dict, ast.Dict):
            components = {}
            for key_node, value_node in zip(maybe_dict.keys, maybe_dict.values):
                if key_node is None:
                    continue
                if isinstance(key_node, ast.Constant):
                    key = str(key_node.value)
                else:
                    key = _safe_unparse(key_node)
                components[key] = _safe_unparse(value_node)
            return components
    return {}


def extract_component_manifest(code: str) -> List[Dict[str, str]]:
    """Extract every returned reward component and its formula/expression."""
    assignments = _extract_assignments_with_conditions(code)
    returned_components = _find_return_component_dict(code)
    manifest: List[Dict[str, str]] = []

    if returned_components:
        for component, returned_expr in returned_components.items():
            assignment_rows = assignments.get(returned_expr, [])
            if assignment_rows:
                formula = ' ; '.join([f'if {cond}: {expr}' for cond, expr in assignment_rows])
            else:
                formula = returned_expr
            manifest.append({
                'component': component,
                'returned_expression': returned_expr,
                'formula': formula,
            })
        return manifest

    # Fallback: expose likely reward variables if the function did not return a
    # component dictionary. This should be rare because coding instructions ask
    # for individual components.
    for name, rows in assignments.items():
        if 'reward' in name or 'bonus' in name or 'penalty' in name or 'cost' in name:
            formula = ' ; '.join([f'if {cond}: {expr}' for cond, expr in rows])
            manifest.append({
                'component': name,
                'returned_expression': name,
                'formula': formula,
            })
    return manifest


def render_component_manifest(code: str) -> str:
    manifest = extract_component_manifest(code)
    if not manifest:
        return 'Component manifest could not be extracted automatically. Use the full reward code below as the source of truth.'

    lines = [
        '| component | returned expression | formula / condition path |',
        '|---|---|---|',
    ]
    for item in manifest:
        formula = item['formula'].replace('\n', ' ')
        lines.append(f"| {item['component']} | `{item['returned_expression']}` | `{formula}` |")
    return '\n'.join(lines)


def build_elite_chain_summary(history: List[Dict[str, Any]]) -> str:
    elites = [record for record in history if record.get('is_elite')]
    if not elites:
        return 'No elite has been confirmed yet.'

    lines = [
        '| elite | parent | fitness | success_like | episode_length | code_hash |',
        '|---|---|---:|---:|---:|---|',
    ]
    for record in elites:
        summary = record.get('final_eval_summary', {}) or {}
        lines.append(
            '| {label} | {parent} | {fitness} | {success} | {ep_len} | {code_hash} |'.format(
                label=_label(record),
                parent=record.get('parent') or 'None',
                fitness=f"{float(record.get('fitness', 0.0)):.3f}",
                success=_final_metric(summary, 'success_like_rate'),
                ep_len=_final_metric(summary, 'episode_length'),
                code_hash=record.get('code_hash', 'NA'),
            )
        )
    return '\n'.join(lines)


def build_elite_reward_code_section(history: List[Dict[str, Any]]) -> str:
    elites = [record for record in history if record.get('is_elite')]
    if not elites:
        return 'No elite reward code has been confirmed yet.'

    blocks = []
    for record in elites:
        code = record.get('reward_code', '')
        blocks.extend([
            f"## {_label(record)} full elite reward code",
            f"fitness={float(record.get('fitness', 0.0)):.3f}, parent={record.get('parent') or 'None'}, code_hash={record.get('code_hash', 'NA')}",
            '',
            'Component manifest:',
            render_component_manifest(code),
            '',
            '```python',
            code.strip(),
            '```',
            '',
        ])
    return '\n'.join(blocks)


def build_failed_children_summary(history: List[Dict[str, Any]], parent_label: Optional[str]) -> str:
    if not parent_label:
        return 'No parent-specific child search history yet.'

    children = [record for record in history if record.get('parent') == parent_label]
    if not children:
        return f'No recorded children from {parent_label} yet.'

    blocks = [
        f'Children searched from current parent {parent_label}:',
        '| child | status | fitness | success_like | episode_length | code_hash |',
        '|---|---|---:|---:|---:|---|',
    ]
    for record in children:
        summary = record.get('final_eval_summary', {}) or {}
        status = 'elite' if record.get('is_elite') else 'failed_or_non_elite_child'
        blocks.append(
            '| {label} | {status} | {fitness} | {success} | {ep_len} | {code_hash} |'.format(
                label=_label(record),
                status=status,
                fitness=f"{float(record.get('fitness', 0.0)):.3f}",
                success=_final_metric(summary, 'success_like_rate'),
                ep_len=_final_metric(summary, 'episode_length'),
                code_hash=record.get('code_hash', 'NA'),
            )
        )

    blocks.append('')
    blocks.append('Child component manifests. These are included to avoid repeating failed reward designs without sending noisy raw evidence:')
    for record in children:
        code = record.get('reward_code', '')
        blocks.extend([
            '',
            f"## {_label(record)} component manifest, status={'elite' if record.get('is_elite') else 'failed_or_non_elite_child'}",
            render_component_manifest(code),
        ])
    return '\n'.join(blocks)


def build_duplicate_guard(history: List[Dict[str, Any]], parent_label: Optional[str]) -> str:
    hashes = []
    for record in history:
        code_hash = record.get('code_hash')
        if not code_hash:
            continue
        relation = 'child_of_current_parent' if parent_label and record.get('parent') == parent_label else 'history'
        hashes.append(f'- {code_hash}: {_label(record)}, {relation}, fitness={float(record.get("fitness", 0.0)):.3f}')

    if not hashes:
        return 'No previous reward hashes yet.'

    return 'Do not regenerate rewards with these exact normalized code hashes:\n' + '\n'.join(hashes)


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
) -> str:
    """Build expert prompt without raw-noise files and without truncation.

    The prompt excludes raw expert_memory_context, full evals, previous LLM
    responses, and full env code after bootstrap. It preserves all elite reward
    function code and all component formulas/manifests needed for expert analysis.
    """
    blocks = [
        '# Expert reward reflection input',
        'Use only the relevant evidence below. Raw evidence files are intentionally not provided because they add noise. Do not ask for extra files.',
        '',
        '## Task card',
        compact_task_card(task_description, env_code),
        '',
        '## Elite chain summary',
        build_elite_chain_summary(reward_history),
        '',
        '## Full elite reward functions and component formulas',
        build_elite_reward_code_section(reward_history),
        '',
        '## Failed / non-elite children from current parent',
        build_failed_children_summary(reward_history, parent_label),
        '',
        '## Duplicate guard',
        build_duplicate_guard(reward_history, parent_label),
        '',
        '## Current iteration compact component feedback',
        component_feedback.strip(),
        '',
        '## Current parent reward code to modify',
        f'Parent: {parent_label}',
        '',
        'Current parent component manifest:',
        render_component_manifest(parent_reward_code),
        '',
        '```python',
        parent_reward_code.strip(),
        '```',
        '',
        '## Reflection rules',
        reflection_init.strip(),
        reflection_end.strip(),
        '',
        '## Output contract',
        'First output the complete component diagnosis table. Then output exactly one Python code block containing the revised compute_reward function. Do not output or request extra files.',
    ]

    return '\n'.join(blocks) + '\n'

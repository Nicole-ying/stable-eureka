from __future__ import annotations

from pathlib import Path
import sys
import json
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eg_rsa.env_adapters.action_primitive_mapper import ActionPrimitiveMapper
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer
from eg_rsa.reward.safe_compiler import SafeRewardCompiler
from eg_rsa.training.schema_reward_wrapper import SchemaRewardWrapper


def assert_close(a: float, b: float, name: str, tol: float = 1e-9) -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{name} mismatch: wrapper={a}, compiled={b}")


def build_schema() -> RewardSchema:
    primitive = {
        "allowed_formula_variables": [
            "x",
            "y",
            "vx",
            "vy",
            "angle",
            "angular_velocity",
            "left_contact",
            "right_contact",
            "contact",
            "both_contact",
            "main_engine",
            "side_engine",
        ],
        "allowed_formula_functions": [
            "abs",
            "min",
            "max",
            "clip",
            "sqrt",
            "exp",
            "tanh",
        ],
        "semantic_roles": [
            "dense_guidance",
            "stability_quality",
            "control_cost",
            "terminal_success",
            "safety_constraint",
        ],
        "action_variables": [
            {"name": "main_engine", "type": "float"},
            {"name": "side_engine", "type": "float"},
        ],
        "action_mapping": {
            "type": "discrete_lookup",
            "variables": {
                "main_engine": {"2": 1.0, "default": 0.0},
                "side_engine": {"1": -1.0, "3": 1.0, "default": 0.0},
            },
        },
    }

    raw_schema = {
        "version": 2,
        "metadata": {
            "source": "wrapper_compiler_equivalence_test",
        },
        "components": [
            {
                "name": "r_progress",
                "type": "formula_component",
                "weight": 1.0,
                "formula": "1.0 - min(abs(x) + 0.5 * abs(vx), 1.0)",
                "params": {
                    "formula": "1.0 - min(abs(x) + 0.5 * abs(vx), 1.0)"
                },
                "clip": [0.0, 1.0],
                "enabled": True,
                "semantic_role": "dense_guidance",
            },
            {
                "name": "r_stability",
                "type": "formula_component",
                "weight": 0.5,
                "formula": "1.0 - min(abs(angle) + abs(angular_velocity), 1.0)",
                "params": {
                    "formula": "1.0 - min(abs(angle) + abs(angular_velocity), 1.0)"
                },
                "clip": [0.0, 1.0],
                "enabled": True,
                "semantic_role": "stability_quality",
            },
            {
                "name": "r_control_cost",
                "type": "formula_component",
                "weight": 0.05,
                "formula": "main_engine + abs(side_engine)",
                "params": {
                    "formula": "main_engine + abs(side_engine)"
                },
                "clip": [-1.0, 0.0],
                "enabled": True,
                "semantic_role": "control_cost",
            },
            {
                "name": "r_low_altitude_bonus",
                "type": "conditional_formula_component",
                "weight": 0.2,
                "formula": "1.0 - min(abs(vy), 1.0)",
                "condition": "y < 0.5",
                "params": {
                    "formula": "1.0 - min(abs(vy), 1.0)",
                    "condition": "y < 0.5",
                },
                "clip": [0.0, 1.0],
                "enabled": True,
                "semantic_role": "dense_guidance",
            },
        ],
        "event_rules": [
            {
                "name": "terminal_success",
                "type": "event_predicate",
                "reward": 100.0,
                "condition": {
                    "expression": "left_contact and right_contact and abs(vy) < 0.3 and abs(angle) < 0.2",
                    "duration_steps": 1,
                },
                "one_time": True,
                "enabled": True,
                "semantic_role": "terminal_success",
            }
        ],
    }

    canonical, report = SchemaCanonicalizer.canonicalize_schema(
        raw_schema,
        primitive_interface=primitive,
        reward_blueprint={},
    )

    if not report.get("ok", False):
        raise AssertionError(f"canonicalization failed: {report}")

    return RewardSchema.from_dict(canonical)


def make_wrapper_harness(schema: RewardSchema):
    obj = object.__new__(SchemaRewardWrapper)
    obj.reward_schema = schema
    obj.action_mapper = ActionPrimitiveMapper(
        mapping_spec=(schema.metadata or {}).get("action_mapping", {}),
        action_variables=(schema.metadata or {}).get("action_variables", []),
    )
    obj._fired_event_rules = set()
    obj._event_rule_duration_counts = {}
    obj._prev_task_metrics = {}
    obj._metric_stagnation_counts = {}
    return obj


def make_compiled_harness(schema: RewardSchema):
    src = "class CompiledRewardHarness:\n" + SafeRewardCompiler.compile(schema)
    namespace = {}
    exec(src, namespace)
    return namespace["CompiledRewardHarness"]()


def compare_once(
    wrapper_obj,
    compiled_obj,
    obs_map,
    action,
    events,
    task_metrics,
    label: str,
) -> None:
    wrapper_total, wrapper_components = wrapper_obj._compute_schema_reward(
        obs_map=obs_map,
        action=action,
        events=events,
        task_metrics=task_metrics,
    )

    compiled_obs_map = dict(obs_map)
    compiled_obs_map["task_metrics"] = dict(task_metrics)
    compiled_total, compiled_components = compiled_obj.compute_reward(
        compiled_obs_map,
        action=action,
        state_flags=events,
    )

    assert_close(wrapper_total, compiled_total, f"{label}.total")

    keys = sorted(set(wrapper_components) | set(compiled_components))
    for key in keys:
        assert key in wrapper_components, f"{label}: missing wrapper component {key}"
        assert key in compiled_components, f"{label}: missing compiled component {key}"
        assert_close(
            wrapper_components[key],
            compiled_components[key],
            f"{label}.{key}",
        )


def main() -> None:
    schema = build_schema()

    # Case 1: no contact, action=2 should map main_engine=1, side_engine=0.
    wrapper_1 = make_wrapper_harness(schema)
    compiled_1 = make_compiled_harness(schema)
    compare_once(
        wrapper_obj=wrapper_1,
        compiled_obj=compiled_1,
        obs_map={
            "x": 0.2,
            "y": 0.8,
            "vx": -0.1,
            "vy": -0.2,
            "angle": 0.05,
            "angular_velocity": 0.02,
            "left_contact": False,
            "right_contact": False,
            "contact": False,
            "both_contact": False,
        },
        action=2,
        events={
            "left_contact": False,
            "right_contact": False,
            "contact": False,
            "both_contact": False,
        },
        task_metrics={},
        label="case_no_contact_main_engine",
    )

    # Case 2: low altitude, side engine action=1 should map side_engine=-1.
    wrapper_2 = make_wrapper_harness(schema)
    compiled_2 = make_compiled_harness(schema)
    compare_once(
        wrapper_obj=wrapper_2,
        compiled_obj=compiled_2,
        obs_map={
            "x": -0.1,
            "y": 0.3,
            "vx": 0.05,
            "vy": -0.1,
            "angle": 0.1,
            "angular_velocity": 0.03,
            "left_contact": False,
            "right_contact": False,
            "contact": False,
            "both_contact": False,
        },
        action=1,
        events={
            "left_contact": False,
            "right_contact": False,
            "contact": False,
            "both_contact": False,
        },
        task_metrics={},
        label="case_low_altitude_side_engine",
    )

    # Case 3: terminal event is one_time. First call pays, second call should not.
    wrapper_3 = make_wrapper_harness(schema)
    compiled_3 = make_compiled_harness(schema)

    terminal_obs = {
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": -0.1,
        "angle": 0.05,
        "angular_velocity": 0.0,
        "left_contact": True,
        "right_contact": True,
        "contact": True,
        "both_contact": True,
    }
    terminal_events = {
        "left_contact": True,
        "right_contact": True,
        "contact": True,
        "both_contact": True,
    }

    compare_once(
        wrapper_obj=wrapper_3,
        compiled_obj=compiled_3,
        obs_map=terminal_obs,
        action=0,
        events=terminal_events,
        task_metrics={},
        label="case_terminal_first_call",
    )

    compare_once(
        wrapper_obj=wrapper_3,
        compiled_obj=compiled_3,
        obs_map=terminal_obs,
        action=0,
        events=terminal_events,
        task_metrics={},
        label="case_terminal_second_call_one_time",
    )

    print(json.dumps({
        "ok": True,
        "message": "SchemaRewardWrapper and SafeRewardCompiler are semantically equivalent for tested canonical schema cases.",
        "tested_cases": [
            "case_no_contact_main_engine",
            "case_low_altitude_side_engine",
            "case_terminal_first_call",
            "case_terminal_second_call_one_time",
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

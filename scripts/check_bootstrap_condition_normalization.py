from __future__ import annotations

from pathlib import Path
import sys
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eg_rsa.reward.bootstrap_schema_validator import BootstrapSchemaValidator
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.safe_compiler import SafeRewardCompiler


def main() -> None:
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
        "metadata": {"source": "condition_normalization_test"},
        "components": [
            {
                "name": "r_safe_descent",
                "type": "conditional_formula_component",
                "weight": 1.0,
                "formula": "1.0 - min(abs(vy), 1.0)",
                "condition": "y < 0.5 && !both_contact",
                "semantic_role": "dense_guidance",
                "enabled": True,
            },
            {
                "name": "r_control_cost",
                "type": "action_penalty",
                "weight": 0.05,
                "formula": "main_engine + abs(side_engine)",
                "semantic_role": "control_cost",
                "enabled": True,
            },
        ],
        "event_rules": [
            {
                "name": "terminal_landing",
                "type": "event_predicate",
                "reward": 100.0,
                "condition": {
                    "expression": "left_contact && right_contact && abs(vy) < 0.3 && abs(angle) < 0.2",
                    "duration_steps": 1,
                },
                "one_time": True,
                "enabled": True,
            },
            {
                "name": "crash",
                "type": "event_predicate",
                "reward": -50.0,
                "expression": "y <= 0 && !(left_contact && right_contact)",
                "condition": {"duration_steps": 1},
                "one_time": True,
                "enabled": True,
            },
        ],
    }

    canonical, report = SchemaCanonicalizer.canonicalize_schema(
        raw_schema,
        primitive_interface=primitive,
        reward_blueprint={},
    )

    validation = BootstrapSchemaValidator.validate_schema(
        canonical,
        primitive_interface=primitive,
        reward_blueprint={},
    )

    if not validation.ok:
        raise AssertionError(json.dumps(validation.to_dict(), indent=2, ensure_ascii=False))

    rules = {r["name"]: r for r in canonical["event_rules"]}
    comps = {c["name"]: c for c in canonical["components"]}

    assert "&&" not in rules["terminal_landing"]["condition"]["expression"]
    assert " and " in rules["terminal_landing"]["condition"]["expression"]

    assert "&&" not in rules["crash"]["condition"]["expression"]
    assert " not " in rules["crash"]["condition"]["expression"]

    assert "&&" not in comps["r_safe_descent"]["condition"]
    assert " not " in comps["r_safe_descent"]["condition"]

    assert comps["r_control_cost"]["type"] == "formula_component"
    assert comps["r_control_cost"]["semantic_role"] == "control_cost"

    schema = RewardSchema.from_dict(canonical)
    compiled = SafeRewardCompiler.compile(schema)
    assert "terminal_landing" in compiled

    print(json.dumps({
        "ok": True,
        "canonicalization_report": report,
        "validation": validation.to_dict(),
        "terminal_landing_condition": rules["terminal_landing"]["condition"]["expression"],
        "crash_condition": rules["crash"]["condition"]["expression"],
        "component_condition": comps["r_safe_descent"]["condition"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

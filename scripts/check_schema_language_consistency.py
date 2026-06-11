from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer
from eg_rsa.reward.bootstrap_schema_validator import BootstrapSchemaValidator
from eg_rsa.reward.safe_compiler import SafeRewardCompiler


def main() -> None:
    primitive = {
        "allowed_formula_variables": [
            "x", "y", "vx", "vy", "angle", "angular_velocity",
            "left_contact", "right_contact", "main_engine", "side_engine",
            "contact", "both_contact",
        ],
        "allowed_formula_functions": ["abs", "min", "max", "clip", "sqrt", "exp", "tanh"],
        "semantic_roles": [
            "dense_guidance",
            "stability_quality",
            "control_cost",
            "terminal_success",
            "safety_constraint",
        ],
    }

    raw_schema = {
        "version": 2,
        "metadata": {"source": "consistency_check"},
        "components": [
            {
                "name": "fuel_cost",
                "type": "action_penalty",
                "role": "control_cost",
                "weight": 0.05,
                "formula": "main_engine + abs(side_engine)",
                "params": {"formula": "main_engine + abs(side_engine)"},
                "enabled": True,
            }
        ],
        "event_rules": [
            {
                "name": "successful_landing",
                "type": "event_predicate",
                "reward": 100.0,
                "condition": {
                    "expression": "left_contact and right_contact",
                    "duration_steps": 1,
                },
                "one_time": True,
                "enabled": True,
            },
            {
                "name": "crash",
                "type": "event_predicate",
                "reward": -50.0,
                "expression": "(y <= 0) and (not left_contact or not right_contact)",
                "condition": {"duration_steps": 1},
                "one_time": True,
                "enabled": True,
            },
        ],
    }

    canonical, report = SchemaCanonicalizer.canonicalize_schema(raw_schema, primitive)

    comp = canonical["components"][0]
    assert comp["type"] == "formula_component", comp
    assert comp["semantic_role"] == "control_cost", comp
    assert comp["params"]["formula"].startswith("-abs("), comp

    rules = {r["name"]: r for r in canonical["event_rules"]}
    assert rules["successful_landing"]["weight"] == 100.0, rules["successful_landing"]
    assert "reward" not in rules["successful_landing"], rules["successful_landing"]
    assert rules["crash"]["weight"] == -50.0, rules["crash"]
    assert rules["crash"]["condition"]["expression"], rules["crash"]

    validation = BootstrapSchemaValidator.validate_schema(canonical, primitive, reward_blueprint={})
    assert validation.ok, validation.to_dict()

    schema = RewardSchema.from_dict(canonical)
    SafeRewardCompiler.compile(schema)

    print(json.dumps({"ok": True, "canonicalization_report": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

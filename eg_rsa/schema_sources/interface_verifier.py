from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple


class InterfaceContractVerifier:
    """Verify and canonicalize LLM-generated primitive_interface.

    Core idea:
      - LLM can infer candidate environment semantics.
      - System owns the final executable interface contract.

    This prevents the common failure:
      LLM says a variable is allowed, but SchemaRewardWrapper cannot provide it
      at runtime.
    """

    @classmethod
    def verify_and_canonicalize(
        cls,
        primitive_interface: Dict[str, Any],
        expected_obs_dim: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = copy.deepcopy(primitive_interface or {})
        errors: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []

        obs_vars = cls._normalize_variable_list(data.get("observation_variables"))
        action_vars = cls._normalize_variable_list(data.get("action_variables"))

        if not obs_vars:
            errors.append("primitive_interface.observation_variables is empty or invalid")

        data["observation_variables"] = obs_vars
        data["action_variables"] = action_vars

        obs_names = [x["name"] for x in obs_vars]

        observation_mapping = cls._canonical_observation_mapping(
            raw_mapping=data.get("observation_mapping"),
            obs_names=obs_names,
            expected_obs_dim=expected_obs_dim,
            warnings=warnings,
            notes=notes,
        )
        data["observation_mapping"] = observation_mapping

        action_mapping = cls._canonical_action_mapping(
            raw_mapping=data.get("action_mapping"),
            action_vars=action_vars,
            warnings=warnings,
            errors=errors,
            notes=notes,
        )
        data["action_mapping"] = action_mapping

        runtime_action_names = cls.runtime_action_variable_names_from_mapping(
            action_mapping=action_mapping,
            action_vars=action_vars,
        )

        declared_action_names = {x["name"] for x in action_vars}
        for name in runtime_action_names:
            if name not in declared_action_names:
                action_vars.append(
                    {
                        "name": name,
                        "description": "Action-derived runtime primitive generated from action_mapping.",
                        "type": "float",
                    }
                )
                declared_action_names.add(name)
                warnings.append(
                    f"action_mapping variable {name!r} was not declared in action_variables; added declaration."
                )

        data["action_variables"] = action_vars

        runtime_allowed = cls._dedupe(obs_names + runtime_action_names)

        raw_allowed = data.get("allowed_formula_variables")
        raw_allowed_list = [str(x) for x in raw_allowed] if isinstance(raw_allowed, list) else []

        dropped = [x for x in raw_allowed_list if x not in runtime_allowed]
        missing = [x for x in runtime_allowed if x not in raw_allowed_list]

        if dropped:
            warnings.append(
                "Dropped allowed_formula_variables that runtime cannot provide: "
                + json.dumps(dropped, ensure_ascii=False)
            )

        if missing:
            notes.append(
                "Added runtime-provided variables to allowed_formula_variables: "
                + json.dumps(missing, ensure_ascii=False)
            )

        # System owns final allowed variables.
        # Do not trust LLM-declared allowed_formula_variables directly.
        data["allowed_formula_variables"] = runtime_allowed

        data.setdefault("version", 1)
        data.setdefault("input_boundary", "anonymous_source_to_primitive_interface")
        data.setdefault("identity_hidden_from_llm", True)

        policy = data.get("bootstrap_interface_policy")
        if not isinstance(policy, dict):
            policy = {}

        policy.update(
            {
                "contract_verified_by_system": True,
                "allowed_variables_source": "system_canonical_runtime_contract",
                "formula_boundary": "formula_ast may use only canonical allowed_formula_variables",
                "raw_action_variable_policy": (
                    "Raw discrete action id is not automatically exposed. "
                    "Use action_mapping output variables such as main_engine/side_engine."
                ),
            }
        )
        data["bootstrap_interface_policy"] = policy

        report = {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "notes": notes,
            "canonical_allowed_formula_variables": data.get("allowed_formula_variables", []),
            "observation_mapping": data.get("observation_mapping", {}),
            "action_mapping": data.get("action_mapping", {}),
            "runtime_action_variables": runtime_action_names,
        }

        return data, report

    @classmethod
    def runtime_formula_variables(cls, primitive_interface: Dict[str, Any]) -> set[str]:
        primitive_interface = primitive_interface or {}

        obs_vars = cls._normalize_variable_list(primitive_interface.get("observation_variables"))
        action_vars = cls._normalize_variable_list(primitive_interface.get("action_variables"))
        action_mapping = primitive_interface.get("action_mapping")

        obs_names = [x["name"] for x in obs_vars]
        action_names = cls.runtime_action_variable_names_from_mapping(action_mapping, action_vars)

        return set(obs_names + action_names)

    @classmethod
    def runtime_action_variable_names_from_mapping(
        cls,
        action_mapping: Any,
        action_vars: List[Dict[str, Any]],
    ) -> List[str]:
        action_mapping = action_mapping if isinstance(action_mapping, dict) else {}
        mapping_type = str(action_mapping.get("type", "") or "").lower()
        mapping_vars = action_mapping.get("variables", {})

        if mapping_type in {"discrete_lookup", "continuous_indices"} and isinstance(mapping_vars, dict):
            return cls._dedupe([str(k) for k in mapping_vars.keys() if str(k)])

        return cls._dedupe(
            [
                str(item["name"])
                for item in action_vars
                if isinstance(item, dict) and item.get("name")
            ]
        )

    @classmethod
    def _canonical_observation_mapping(
        cls,
        raw_mapping: Any,
        obs_names: List[str],
        expected_obs_dim: Optional[int],
        warnings: List[str],
        notes: List[str],
    ) -> Dict[str, int]:
        raw_mapping = raw_mapping if isinstance(raw_mapping, dict) else {}

        if not raw_mapping:
            notes.append("observation_mapping missing; filled by observation_variables order.")

        out: Dict[str, int] = {}
        used: set[int] = set()

        for default_idx, name in enumerate(obs_names):
            raw_idx = raw_mapping.get(name, default_idx)

            try:
                idx = int(raw_idx)
            except Exception:
                idx = default_idx
                warnings.append(f"observation_mapping[{name!r}] is not int; replaced with {default_idx}.")

            if idx < 0:
                warnings.append(f"observation_mapping[{name!r}] is negative; replaced with {default_idx}.")
                idx = default_idx

            if expected_obs_dim is not None and idx >= int(expected_obs_dim):
                warnings.append(
                    f"observation_mapping[{name!r}]={idx} exceeds expected_obs_dim={expected_obs_dim}; "
                    f"replaced with {default_idx}."
                )
                idx = default_idx

            if idx in used:
                warnings.append(
                    f"observation_mapping index {idx} is reused; keeping it, but check whether this is intended."
                )

            used.add(idx)
            out[name] = idx

        extra = sorted(str(k) for k in raw_mapping.keys() if str(k) not in set(obs_names))
        if extra:
            warnings.append(
                "Ignored observation_mapping keys not declared in observation_variables: "
                + json.dumps(extra, ensure_ascii=False)
            )

        return out

    @classmethod
    def _canonical_action_mapping(
        cls,
        raw_mapping: Any,
        action_vars: List[Dict[str, Any]],
        warnings: List[str],
        errors: List[str],
        notes: List[str],
    ) -> Dict[str, Any]:
        mapping = copy.deepcopy(raw_mapping) if isinstance(raw_mapping, dict) else {}

        if not mapping:
            notes.append(
                "action_mapping missing or empty; runtime will fall back to declared action_variables/action_i."
            )
            return {}

        mapping_type = str(mapping.get("type", "") or "").lower()
        variables = mapping.get("variables", {})

        if mapping_type == "discrete_lookup":
            if not isinstance(variables, dict) or not variables:
                errors.append("action_mapping.type=discrete_lookup requires non-empty variables dict")
                return {"type": "discrete_lookup", "variables": {}}

            canonical_vars: Dict[str, Dict[str, float]] = {}

            for name, table in variables.items():
                name = str(name)
                if not name:
                    warnings.append("Ignored empty action_mapping variable name.")
                    continue

                if not isinstance(table, dict):
                    warnings.append(f"discrete_lookup table for {name!r} is not dict; using default 0.0.")
                    canonical_vars[name] = {"default": 0.0}
                    continue

                out_table: Dict[str, float] = {}
                for key, value in table.items():
                    try:
                        out_table[str(key)] = float(value)
                    except Exception:
                        warnings.append(
                            f"discrete_lookup value for variable {name!r}, key {key!r} is non-numeric; ignored."
                        )

                out_table.setdefault("default", 0.0)
                canonical_vars[name] = out_table

            return {"type": "discrete_lookup", "variables": canonical_vars}

        if mapping_type == "continuous_indices":
            if not isinstance(variables, dict) or not variables:
                errors.append("action_mapping.type=continuous_indices requires non-empty variables dict")
                return {"type": "continuous_indices", "variables": {}}

            canonical_vars: Dict[str, int] = {}

            for name, idx in variables.items():
                name = str(name)

                try:
                    idx_int = int(idx)
                except Exception:
                    warnings.append(f"continuous_indices index for {name!r} is not int; ignored.")
                    continue

                if idx_int < 0:
                    warnings.append(f"continuous_indices index for {name!r} is negative; ignored.")
                    continue

                canonical_vars[name] = idx_int

            return {"type": "continuous_indices", "variables": canonical_vars}

        warnings.append(
            f"Unsupported action_mapping.type={mapping_type!r}; runtime will fall back to action_variables/action_i."
        )
        return {}

    @staticmethod
    def _normalize_variable_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for item in value:
            if not isinstance(item, dict) or not item.get("name"):
                continue

            name = str(item["name"]).strip()
            if not name or name in seen:
                continue

            seen.add(name)

            typ = str(item.get("type", "float")).lower()
            out.append(
                {
                    "name": name,
                    "description": str(item.get("description", "")),
                    "type": "bool" if typ in {"bool", "boolean"} else "float",
                }
            )

        return out

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()

        for value in values:
            value = str(value)
            if value and value not in seen:
                out.append(value)
                seen.add(value)

        return out

#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Check repo layout..."
test -f train_eg_rsa.py
test -d eg_rsa
test -f eg_rsa/llm/bootstrap_agent.py
test -f eg_rsa/schema_sources/llm_bootstrap.py
test -f eg_rsa/reward/bootstrap_schema_validator.py
test -f eg_rsa/training/eg_rsa_trainer.py

echo "[2/6] Backup files..."
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".eg_rsa_patch_backup_${TS}"

cp eg_rsa/llm/bootstrap_agent.py ".eg_rsa_patch_backup_${TS}/bootstrap_agent.py"
cp eg_rsa/schema_sources/llm_bootstrap.py ".eg_rsa_patch_backup_${TS}/llm_bootstrap.py"
cp eg_rsa/reward/bootstrap_schema_validator.py ".eg_rsa_patch_backup_${TS}/bootstrap_schema_validator.py"
cp eg_rsa/training/eg_rsa_trainer.py ".eg_rsa_patch_backup_${TS}/eg_rsa_trainer.py"

echo "[3/6] Add InterfaceContractVerifier..."
cat > eg_rsa/schema_sources/interface_verifier.py <<'PY'
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
PY

echo "[4/6] Patch BootstrapAgent: add Env-Interface Agent..."
python - <<'PY'
from pathlib import Path

p = Path("eg_rsa/llm/bootstrap_agent.py")
text = p.read_text(encoding="utf-8")

needle = '''    def generate_bootstrap_from_source(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Infer primitive interface and initial schema from anonymous source input."""
        prompt = self._build_source_prompt(task_spec)
        self.last_prompt = prompt

        if self.llm_client is None:
            primitive_interface = self._primitive_interface_from_task_spec(task_spec)
            result = self._fallback_bootstrap(primitive_interface, primitive_interface.get("task_description", ""))
            result["primitive_interface"] = primitive_interface
            result.setdefault("bootstrap_report", {})["source_aware_bootstrap"] = True
            result["bootstrap_report"]["used_llm"] = False
            self.last_response_text = json.dumps(result, indent=2, ensure_ascii=False)
            return result

        response_text = self.llm_client.generate(prompt)
        self.last_response_text = response_text
        parsed = extract_json_object(response_text)
        return self._normalize_source_bootstrap(parsed, task_spec)

'''

insert = needle + '''    def generate_primitive_interface_from_source(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Env-Interface Agent.

        This stage reads anonymous task/source input and proposes only a candidate
        primitive_interface. It must not generate reward schema. The system will
        verify/canonicalize the interface before reward bootstrap.
        """
        prompt = self._build_source_interface_prompt(task_spec)
        self.last_prompt = prompt

        if self.llm_client is None:
            primitive_interface = self._primitive_interface_from_task_spec(task_spec)
            result = {
                "source_understanding": {
                    "task_objective": str(task_spec.get("task_description", "")),
                    "observation_semantics": [],
                    "action_semantics": [],
                    "uncertainties": ["fallback interface generated without LLM"],
                },
                "primitive_interface": primitive_interface,
                "interface_report": {
                    "used_llm": False,
                    "notes": ["fallback interface only"],
                },
            }
            self.last_response_text = json.dumps(result, indent=2, ensure_ascii=False)
            return result

        response_text = self.llm_client.generate(prompt)
        self.last_response_text = response_text
        parsed = extract_json_object(response_text)

        if not isinstance(parsed, dict):
            raise ValueError("Interface LLM response must parse to a JSON object")

        primitive_interface = parsed.get("primitive_interface")

        if not isinstance(primitive_interface, dict):
            # Allow the LLM to accidentally return the primitive_interface object directly.
            if "observation_variables" in parsed and "observation_mapping" in parsed:
                primitive_interface = parsed
                parsed = {"primitive_interface": primitive_interface}
            else:
                raise ValueError("Interface LLM response missing primitive_interface")

        primitive_interface = self._normalize_primitive_interface(primitive_interface, task_spec)
        parsed["primitive_interface"] = primitive_interface
        parsed.setdefault("source_understanding", {})
        parsed.setdefault("interface_report", {})
        parsed["interface_report"].setdefault("used_llm", True)
        parsed["interface_report"].setdefault("stage", "env_interface_agent")

        return parsed

'''

if needle not in text:
    raise SystemExit("Could not find generate_bootstrap_from_source block; file may have changed.")

text = text.replace(needle, insert)

needle2 = '''    @staticmethod
    def _build_source_prompt(task_spec: Dict[str, Any]) -> str:
'''

method = '''    @staticmethod
    def _build_source_interface_prompt(task_spec: Dict[str, Any]) -> str:
        visible_source = BootstrapAgent._sanitize_source_task_for_llm(task_spec)

        output_shape = {
            "source_understanding": {
                "task_objective": "objective inferred from task/source text",
                "observation_semantics": [
                    {
                        "index": 0,
                        "name": "short_snake_case",
                        "meaning": "state meaning",
                        "type": "float_or_bool",
                    }
                ],
                "action_semantics": [
                    {
                        "raw_action": 0,
                        "meaning": "control meaning",
                    }
                ],
                "uncertainties": [],
            },
            "primitive_interface": {
                "version": 1,
                "input_boundary": "anonymous_source_to_primitive_interface",
                "identity_hidden_from_llm": True,
                "task_description": "copy task objective without environment name",
                "observation_variables": [
                    {
                        "name": "short_snake_case_obs",
                        "description": "meaning",
                        "type": "float_or_bool",
                    }
                ],
                "action_variables": [
                    {
                        "name": "short_snake_case_action",
                        "description": "meaning",
                        "type": "float",
                    }
                ],
                "observation_mapping": {
                    "short_snake_case_obs": 0,
                },
                "action_mapping": {
                    "type": "discrete_lookup_or_continuous_indices",
                    "variables": {
                        "semantic_action_variable": {
                            "0": 0.0,
                            "1": 1.0,
                            "default": 0.0,
                        }
                    },
                },
                "allowed_formula_variables": [
                    "all observation variable names plus action_mapping output variable names only"
                ],
                "allowed_formula_functions": BootstrapAgent.DEFAULT_FUNCTIONS,
                "semantic_roles": BootstrapAgent.DEFAULT_SEMANTIC_ROLES,
            },
            "interface_report": {
                "design_rationale": "...",
                "assumptions": [],
                "uncertainties": [],
            },
        }

        return f"""
You are the EG-RSA Env-Interface Agent.

Your input is an anonymized task description plus anonymized environment source/source-summary information.
Your job is to infer a candidate primitive_interface only.

Do not generate reward_blueprint.
Do not generate initial_schema.
Do not generate formula_ast.
Do not output Python code.

Critical constraints:
1. The benchmark/environment name is intentionally hidden. Do not guess or mention it.
2. Use only the task description and anonymized source/source-summary information below.
3. Variable names must be short snake_case and must not include benchmark names.
4. observation_mapping maps each observation variable name to its numeric observation index.
5. If actions are discrete and action meanings are provided, use action_mapping.type="discrete_lookup".
6. If actions are continuous, use action_mapping.type="continuous_indices".
7. For discrete action ids, do not expose a raw variable named "action" unless the raw integer id itself has meaningful numeric ordering.
8. Prefer semantic action variables produced by action_mapping, e.g. main_engine, side_engine, brake, torque.
9. allowed_formula_variables must contain only observation variables plus action_mapping output variables.
10. Output JSON only.

Anonymized task/source input:
{json.dumps(visible_source, indent=2, ensure_ascii=False)}

Required output shape:
{json.dumps(output_shape, indent=2, ensure_ascii=False)}
""".strip()

'''

if needle2 not in text:
    raise SystemExit("Could not find _build_source_prompt insertion point.")

text = text.replace(needle2, method + needle2)

p.write_text(text, encoding="utf-8")
PY

echo "[5/6] Patch LLMBootstrapSchemaSource, Validator, Trainer..."
python - <<'PY'
from pathlib import Path

# ---------------------------------------------------------------------
# Patch eg_rsa/schema_sources/llm_bootstrap.py
# ---------------------------------------------------------------------
p = Path("eg_rsa/schema_sources/llm_bootstrap.py")
text = p.read_text(encoding="utf-8")

if "from eg_rsa.schema_sources.interface_verifier import InterfaceContractVerifier" not in text:
    text = text.replace(
        "from eg_rsa.schema_sources.eureka_like_interface import EurekaLikeInterfaceBuilder\n",
        "from eg_rsa.schema_sources.eureka_like_interface import EurekaLikeInterfaceBuilder\n"
        "from eg_rsa.schema_sources.interface_verifier import InterfaceContractVerifier\n",
    )

needle = '''        else:
            primitive_interface = self._load_or_generate_primitive_interface(cfg)

        runtime_spec = self._build_runtime_spec_from_primitive_interface(primitive_interface)
'''

replacement = '''        else:
            primitive_interface = self._load_or_generate_primitive_interface(cfg)

        primitive_interface, interface_contract_report = InterfaceContractVerifier.verify_and_canonicalize(
            primitive_interface
        )
        self._write_json(bootstrap_dir / "interface_contract_report.json", interface_contract_report)

        if bootstrap_result is not None:
            bootstrap_result["primitive_interface"] = primitive_interface

        if interface_contract_report.get("errors"):
            raise ValueError(
                "Primitive interface failed executable contract verification: "
                + json.dumps(interface_contract_report.get("errors", []), ensure_ascii=False)
            )

        runtime_spec = self._build_runtime_spec_from_primitive_interface(primitive_interface)
'''

if needle not in text:
    raise SystemExit("Could not find primitive_interface verification insertion point in llm_bootstrap.py")

text = text.replace(needle, replacement)

old = '''    def _run_source_aware_bootstrap(self, cfg: Dict[str, Any], bootstrap_dir: Path) -> Dict[str, Any]:
        task_path = cfg.get("eureka_like_task_path") or cfg.get("task_file_path") or cfg.get("source_task_path")
        if not task_path:
            raise ValueError("source-aware bootstrap requires eureka_like_task_path, task_file_path, or source_task_path")
        task_spec = self._load_task_file(Path(str(task_path)))
        result = self.bootstrap_agent.generate_bootstrap_from_source(task_spec)
        primitive_interface = result.get("primitive_interface")
        if not isinstance(primitive_interface, dict):
            raise ValueError("source-aware bootstrap returned no primitive_interface")
        interface_dir = self.output_dir / cfg.get("interface_output_subdir", "interface")
        self._write_json(interface_dir / "anonymous_source_input.json", task_spec)
        self._write_json(interface_dir / "generated_primitive_interface.json", primitive_interface)
        self._write_json(
            interface_dir / "interface_generation_report.json",
            {
                "source": "source_aware_bootstrap_agent",
                "source_path": str(task_path),
                "output_path": str(interface_dir / "generated_primitive_interface.json"),
                "identity_hidden_from_llm": True,
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "notes": [
                    "BootstrapAgent inferred the primitive interface and initial schema in one LLM call.",
                    "The runtime environment name is not included in the bootstrap prompt.",
                    "The primitive interface is an internal audit artifact, not the user-facing entry point.",
                ],
            },
        )
        self.config.setdefault("eg_rsa", {}).setdefault("schema_source", {})[
            "generated_primitive_interface_path"
        ] = str(interface_dir / "generated_primitive_interface.json")
        return result
'''

new = '''    def _run_source_aware_bootstrap(self, cfg: Dict[str, Any], bootstrap_dir: Path) -> Dict[str, Any]:
        """Two-stage source-aware bootstrap.

        Stage 1:
            LLM Env-Interface Agent proposes candidate primitive_interface from
            anonymous task/source input.

        Stage 2:
            System verifies/canonicalizes the interface contract, then LLM Reward
            Bootstrap Agent generates reward_blueprint + initial_schema under the
            fixed executable interface.
        """
        task_path = cfg.get("eureka_like_task_path") or cfg.get("task_file_path") or cfg.get("source_task_path")

        if not task_path:
            raise ValueError("source-aware bootstrap requires eureka_like_task_path, task_file_path, or source_task_path")

        task_spec = self._load_task_file(Path(str(task_path)))
        interface_dir = self.output_dir / cfg.get("interface_output_subdir", "interface")

        interface_result = self.bootstrap_agent.generate_primitive_interface_from_source(task_spec)
        interface_prompt = self.bootstrap_agent.last_prompt
        interface_response = self.bootstrap_agent.last_response_text

        candidate_interface = interface_result.get("primitive_interface")
        if not isinstance(candidate_interface, dict):
            raise ValueError("Env-Interface Agent returned no primitive_interface")

        primitive_interface, interface_contract_report = InterfaceContractVerifier.verify_and_canonicalize(
            candidate_interface
        )

        self._write_json(interface_dir / "anonymous_source_input.json", task_spec)
        self._write_text(interface_dir / "interface_generation_prompt.txt", interface_prompt)
        self._write_text(interface_dir / "interface_generation_response.txt", interface_response)
        self._write_json(interface_dir / "interface_generation_response.json", interface_result)
        self._write_json(interface_dir / "candidate_primitive_interface.json", candidate_interface)
        self._write_json(interface_dir / "generated_primitive_interface.json", primitive_interface)
        self._write_json(interface_dir / "interface_contract_report.json", interface_contract_report)

        if interface_contract_report.get("errors"):
            raise ValueError(
                "Candidate primitive_interface failed executable contract verification: "
                + json.dumps(interface_contract_report.get("errors", []), ensure_ascii=False)
            )

        self._write_json(
            interface_dir / "interface_generation_report.json",
            {
                "source": "two_stage_source_aware_env_interface_agent",
                "source_path": str(task_path),
                "candidate_output_path": str(interface_dir / "candidate_primitive_interface.json"),
                "canonical_output_path": str(interface_dir / "generated_primitive_interface.json"),
                "contract_report_path": str(interface_dir / "interface_contract_report.json"),
                "identity_hidden_from_llm": True,
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "notes": [
                    "Stage 1 LLM inferred a candidate primitive interface from anonymized source/task input.",
                    "System InterfaceContractVerifier canonicalized allowed_formula_variables from runtime-executable variables.",
                    "Stage 2 LLM generated initial_schema only under this verified primitive interface.",
                    "The runtime environment name is not included in the bootstrap prompt.",
                ],
            },
        )

        self.config.setdefault("eg_rsa", {}).setdefault("schema_source", {})[
            "generated_primitive_interface_path"
        ] = str(interface_dir / "generated_primitive_interface.json")

        task_description = primitive_interface.get("task_description", "") or task_spec.get("task_description", "")

        result = self.bootstrap_agent.generate_bootstrap(
            primitive_interface=primitive_interface,
            task_description=task_description,
        )

        result["primitive_interface"] = primitive_interface
        result["source_understanding"] = interface_result.get("source_understanding", {})
        result.setdefault("bootstrap_report", {})
        result["bootstrap_report"].update(
            {
                "source_aware_bootstrap": True,
                "primitive_interface_generated": True,
                "two_stage_bootstrap": True,
                "interface_contract_verified": True,
                "interface_contract_report": interface_contract_report,
            }
        )

        return result
'''

if old not in text:
    raise SystemExit("Could not find _run_source_aware_bootstrap block to replace.")

text = text.replace(old, new)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# Patch eg_rsa/reward/bootstrap_schema_validator.py
# ---------------------------------------------------------------------
p = Path("eg_rsa/reward/bootstrap_schema_validator.py")
text = p.read_text(encoding="utf-8")

if "from eg_rsa.schema_sources.interface_verifier import InterfaceContractVerifier" not in text:
    text = text.replace(
        "from eg_rsa.reward.formula_ast import validate_formula_ast\n",
        "from eg_rsa.reward.formula_ast import validate_formula_ast\n"
        "from eg_rsa.schema_sources.interface_verifier import InterfaceContractVerifier\n",
    )

needle = '''        allowed_vars = set(primitive_interface.get("allowed_formula_variables", []))
        semantic_roles = set(primitive_interface.get("semantic_roles", []))
'''

replacement = '''        declared_allowed_vars = set(str(x) for x in primitive_interface.get("allowed_formula_variables", []))
        runtime_vars = InterfaceContractVerifier.runtime_formula_variables(primitive_interface)

        unavailable = sorted(declared_allowed_vars - runtime_vars)
        if unavailable:
            errors.append(
                "primitive_interface.allowed_formula_variables contains variables that runtime cannot provide: "
                f"{unavailable}"
            )

        allowed_vars = declared_allowed_vars & runtime_vars if declared_allowed_vars else runtime_vars
        semantic_roles = set(primitive_interface.get("semantic_roles", []))
'''

if needle not in text:
    raise SystemExit("Could not find allowed_vars block in bootstrap_schema_validator.py")

text = text.replace(needle, replacement)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# Patch eg_rsa/training/eg_rsa_trainer.py
# ---------------------------------------------------------------------
p = Path("eg_rsa/training/eg_rsa_trainer.py")
text = p.read_text(encoding="utf-8")

needle = '''        n_envs = int(self.config["rl"]["training"].get("n_envs", 1))
        train_env = self._make_training_vec_env(reward_schema, n_envs)
'''

replacement = '''        self._smoke_test_schema_reward(reward_schema)

        n_envs = int(self.config["rl"]["training"].get("n_envs", 1))
        train_env = self._make_training_vec_env(reward_schema, n_envs)
'''

if needle not in text:
    raise SystemExit("Could not find train_and_record insertion point in eg_rsa_trainer.py")

text = text.replace(needle, replacement)

needle2 = '''    def _make_env(self, reward_schema: RewardSchema):
'''

method = '''    def _smoke_test_schema_reward(self, reward_schema: RewardSchema) -> None:
        """Run one real env.step before PPO training.

        This catches schema/runtime contract errors early, e.g. formula_ast uses
        a variable that SchemaRewardWrapper._primitive_vars() cannot provide.
        """
        env = self._make_env(reward_schema)

        try:
            seed = self.config["rl"]["training"].get("seed", None)
            env.reset(seed=seed)
            action = env.action_space.sample()
            env.step(action)
        except Exception as exc:
            raise RuntimeError(
                "Schema reward runtime smoke test failed before PPO training. "
                "The initial_schema likely uses variables that are not available "
                "in the verified primitive_interface/runtime variable table."
            ) from exc
        finally:
            env.close()

'''

if needle2 not in text:
    raise SystemExit("Could not find _make_env insertion point in eg_rsa_trainer.py")

text = text.replace(needle2, method + needle2)
p.write_text(text, encoding="utf-8")
PY

echo "[6/6] Syntax check..."
python -m py_compile \
  eg_rsa/schema_sources/interface_verifier.py \
  eg_rsa/llm/bootstrap_agent.py \
  eg_rsa/schema_sources/llm_bootstrap.py \
  eg_rsa/reward/bootstrap_schema_validator.py \
  eg_rsa/training/eg_rsa_trainer.py

echo ""
echo "Patch done."
echo "Backups saved in: .eg_rsa_patch_backup_${TS}"
echo ""
echo "Next:"
echo "  rm -rf experiments/eg_rsa_landing_v2_1_source_aware_bootstrap_check"
echo "  python train_eg_rsa.py --config configs/eg_rsa_landing_v2_1_source_aware_bootstrap_check.yml"

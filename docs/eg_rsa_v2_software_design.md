# EG-RSA V2 Software Design Document

## 1. Version Positioning

### 1.1 V1 Definition

EG-RSA-V1 is the human-seeded constrained reward self-evolution version.

V1 uses:

    human-written initial reward schema
    human-written diagnostic spec
    human-written task description
    existing reward schema compiler
    existing semantic outcome analyzer
    existing attribution analyzer
    existing memory / lesson store
    existing LLM edit agent
    existing edit validator
    existing scale audit and behavior risk audit

V1 is used as a controlled mechanism-verification setting.

V1 should not be described as fully autonomous reward discovery from scratch.

### 1.2 V2 Definition

EG-RSA-V2 is the LLM-bootstrapped reward self-evolution version.

V2 adds a bootstrap stage before the existing EG-RSA loop:

    primitive task interface
    -> LLM bootstrap agent
    -> generated initial reward schema
    -> generated diagnostic predicates / diagnostic spec
    -> existing EG-RSA self-evolution loop

V2 removes the strongest V1 manual prior: the manually written initial reward schema.

V2 does not remove safety constraints. Instead, it increases LLM freedom through validated schema synthesis.

---

## 2. Design Goal

The goal of V2 is to move from:

    human-seeded constrained schema editing

to:

    LLM-bootstrapped reward self-evolution

The system should support both V1 and V2.

A V1 config must still work exactly as before.

A V2 config enables bootstrap and uses the generated schema as the starting point.

---

## 3. Compatibility Requirement

This is the most important engineering rule.

The runner must satisfy:

    if eg_rsa.bootstrap.enabled == true:
        run V2 bootstrap path
    else:
        run original V1 path

Therefore, any modification to runner.py must be backward-compatible.

V1 config:

    configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml

should continue to use:

    eg_rsa.initial_schema_path

V2 config:

    configs/eg_rsa_deepseek_v2_bootstrap_smoke.yml
    configs/eg_rsa_deepseek_v2_bootstrap_10x1m.yml

should use:

    eg_rsa.bootstrap.enabled: true
    eg_rsa.bootstrap.primitive_interface_path

---

## 4. Current Runner Flow

Current runner flow:

    EGRSARunner.__init__
        load config
        create output_dir
        create ExperimentMode
        build llm client
        create ReflectionAgent
        create EditAgent
        create StructuralSearchAgent
        create AgentActionController
        load structural context

    EGRSARunner.run
        load initial schema from eg_rsa.initial_schema_path
        create memory store
        create lesson store
        for each iteration:
            write reward_schema.json
            compile reward
            train PPO
            collect trajectories
            analyze reward attribution
            analyze semantic outcome
            diagnose failure modes
            inspect trajectories
            retrieve memory and lessons
            decide continue/edit
            generate LLM edit plan
            validate edit plan
            run candidate evaluator / edit gate
            run scale audit
            run behavior risk audit
            apply edit
            update memory / lesson
            rollback if needed
        write summary

V2 only changes the initial schema loading stage.

---

## 5. V2 Target Flow

V2 target flow:

    EGRSARunner.run
        if bootstrap.enabled:
            schema = self._load_or_bootstrap_schema()
        else:
            schema = self._load_schema(initial_schema_path)

        continue original loop unchanged

The bootstrap stage should produce:

    experiments/<output_dir>/bootstrap/bootstrap_prompt.txt
    experiments/<output_dir>/bootstrap/bootstrap_response.json
    experiments/<output_dir>/bootstrap/generated_initial_schema.json
    experiments/<output_dir>/bootstrap/generated_diagnostics.yml
    experiments/<output_dir>/bootstrap/bootstrap_report.json

---

## 6. New Files

### 6.1 eg_rsa/llm/bootstrap_agent.py

Purpose:

    Call LLM to generate initial reward schema and diagnostic spec from a primitive task interface.

Inputs:

    primitive_interface: dict
    existing task_description: optional string
    output constraints: schema format and forbidden terms

Outputs:

    bootstrap_response: dict
    generated_initial_schema: dict
    generated_diagnostics: dict or yaml-compatible dict
    bootstrap_report: dict

Main class:

    BootstrapAgent

Main method:

    generate_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: Optional[str] = None
    ) -> Dict[str, Any]

Expected return:

    {
        "initial_schema": {...},
        "diagnostics": {...},
        "bootstrap_report": {...}
    }

Important prompt constraints:

    Do not use forbidden high-level terms:
        landing_region
        safe_contact
        stable_landing_condition
        approach_region_score
        landing_quality
        stability

    Use only primitive variables:
        x
        y
        vx
        vy
        angle
        angular_velocity
        left_contact
        right_contact
        main_engine
        side_engine

    Assign semantic_role to every component:
        dense_guidance
        stability_quality
        terminal_success
        safety_constraint
        control_cost

    Do not use official environment reward.

---

### 6.2 eg_rsa/reward/bootstrap_schema_validator.py

Purpose:

    Validate LLM-generated initial schema before training.

First version can be simple.

Checks:

    schema is dict
    schema has version
    schema has components list
    schema has event_rules list
    all components have name/type/weight/enabled/semantic_role
    no forbidden terms appear in component names, metric names, condition names
    semantic_role is valid
    component types are supported by current compiler

Important note:

    V2 Patch 2 should not yet introduce arbitrary formula execution.
    Therefore, first bootstrap schema should still use compiler-supported safe types.

Allowed initial types for Patch 2:

    metric_value
    metric_delta
    action_penalty
    event_bonus

But since V2 should avoid pre-defined high-level metrics, Patch 2 may generate only a conservative schema compatible with the existing system, then Patch 3 expands formula support.

If the generated schema uses unsupported types, validation should fail clearly.

---

### 6.3 docs/eg_rsa_v2_software_design.md

This file.

---

## 7. Modified Files

### 7.1 eg_rsa/runner.py

Required changes:

    import BootstrapAgent
    optionally import BootstrapSchemaValidator

Add in __init__:

    self.bootstrap_agent = BootstrapAgent(llm_client=llm_client)

Add method:

    def _load_or_bootstrap_schema(self) -> RewardSchema:
        ...

Pseudo logic:

    bootstrap_cfg = self.config.get("eg_rsa", {}).get("bootstrap", {})
    enabled = bool(bootstrap_cfg.get("enabled", False))

    if not enabled:
        return self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))

    bootstrap_dir = self.output_dir / bootstrap_cfg.get("output_subdir", "bootstrap")
    schema_path = bootstrap_dir / "generated_initial_schema.json"
    reuse = bool(bootstrap_cfg.get("reuse_if_exists", True))

    if reuse and schema_path.exists():
        return self._load_schema(schema_path)

    primitive_interface_path = Path(bootstrap_cfg["primitive_interface_path"])
    primitive_interface = json.loads(primitive_interface_path.read_text())

    task_description = self._load_task_description()

    result = self.bootstrap_agent.generate_bootstrap(
        primitive_interface=primitive_interface,
        task_description=task_description,
    )

    write bootstrap prompt / response / report
    write generated_initial_schema.json
    write generated_diagnostics.yml

    return RewardSchema.from_dict(result["initial_schema"])

Change run():

    old:
        schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))

    new:
        schema = self._load_or_bootstrap_schema()

Backward compatibility:

    If bootstrap.enabled is absent or false, behavior is identical to V1.

---

### 7.2 configs/eg_rsa_deepseek_v2_bootstrap_smoke.yml

Already added.

Purpose:

    quick V2 smoke test
    3 iterations x 100k steps

Important:

    This config should not be treated as final paper experiment.
    It is only for checking bootstrap and training loop.

---

### 7.3 configs/eg_rsa_deepseek_v2_bootstrap_10x1m.yml

Already added.

Purpose:

    formal V2 main experiment
    10 iterations x 1M steps

Do not run until Patch 2, Patch 3, and Patch 4 are stable.

---

## 8. V2 Patch Roadmap

### Patch 2: BootstrapAgent + Runner Insertion

Goal:

    Make V2 bootstrap stage executable.

Scope:

    add BootstrapAgent
    add bootstrap schema validation
    modify runner.py
    write bootstrap artifacts
    preserve V1 compatibility

Not included:

    arbitrary formula components
    new event predicate compiler
    AST formula validator
    signal evaluator
    sandbox short-run

Validation:

    V1 config still starts normally
    V2 smoke config creates bootstrap artifacts
    generated schema compiles
    training starts

---

### Patch 3: Formula / Predicate Schema Synthesis

Goal:

    Increase LLM freedom.

Add component types:

    formula_component
    conditional_formula_component
    event_predicate

Add operators:

    add_formula_component
    add_conditional_formula_component
    add_event_predicate
    modify_component_formula
    modify_event_condition

Compiler support:

    safe expression evaluator
    allowed variables only
    allowed functions only

---

### Patch 4: FormulaValidator + SignalEvaluator

Goal:

    Prevent unsafe or useless LLM-generated reward expressions.

Checks:

    AST whitelist
    variable whitelist
    function whitelist
    finite output
    non-constant signal
    non-zero signal
    active rate
    range / scale
    reward hacking risk

---

### Patch 5: V2 Smoke Test

Command:

    python train_eg_rsa.py --config configs/eg_rsa_deepseek_v2_bootstrap_smoke.yml

Expected outputs:

    bootstrap/generated_initial_schema.json
    bootstrap/generated_diagnostics.yml
    iteration_000/reward_schema.json
    iteration_000/compiled_reward.py
    iteration_000/semantic_outcome.json
    summary.json

Success criteria:

    bootstrap JSON is valid
    schema compiles
    PPO training starts
    no crash in first iteration
    artifacts are written

---

### Patch 6: V2 Full Experiment

Command:

    python train_eg_rsa.py --config configs/eg_rsa_deepseek_v2_bootstrap_10x1m.yml

Success criteria:

    complete 10 iterations
    compare with V1 full
    compare with V2 bootstrap-only frozen
    analyze edit/recovery behavior

---

## 9. V1 Safety Strategy During V2 Development

To prevent V2 changes from breaking V1:

1. Create a Git tag:

        git tag eg-rsa-v1-freeze
        git push origin eg-rsa-v1-freeze

2. Keep V1 config unchanged:

        configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml

3. Add compatibility rule:

        bootstrap.enabled false or missing means old behavior

4. Before every major V2 patch, run a V1 smoke check:

        python train_eg_rsa.py --config configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml

   For quick test, create a temporary copy with:
        iterations = 1
        total_timesteps = 1000

5. Do not modify V1 result directory.

---

## 10. Recommended Next Step

Next implementation step:

    Patch 2: BootstrapAgent + Runner Insertion

Patch 2 should be small and reversible.

It should not yet implement full formula synthesis.

The goal is only:

    primitive interface
    -> LLM-generated initial schema
    -> generated schema loaded by runner
    -> original EG-RSA loop continues

Only after Patch 2 works should Patch 3 expand edit freedom.

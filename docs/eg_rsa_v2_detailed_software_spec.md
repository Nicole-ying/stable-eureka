# EG-RSA V2 Detailed Software Design Specification

## 1. 目标

EG-RSA V2 的目标是从 V1 的：

    人工初始 schema + LLM 受限编辑

升级为：

    LLM 自举 initial schema + LLM 自进化 reward schema search

V2 不应该在主框架里到处写 V1/V2 判断。

Runner 不应该关心版本号。Runner 只关心：

    initial schema 从哪里来？
    reward schema 如何被验证？
    reward schema 如何被编译？
    reward schema 如何被编辑和进化？

因此，V2 采用以下架构：

    SchemaSource
        ManualSchemaSource
        LLMBootstrapSchemaSource

    SchemaValidator
        BootstrapSchemaValidator
        FormulaSchemaValidator

    Reward Compiler
        existing SafeRewardCompiler
        add formula_component support
        add event_predicate support

    Runner
        only calls schema_source.load_or_create()
        then enters the original EG-RSA loop

---

## 2. Git 与版本策略

### 2.1 V1 版本保留

V1 已通过 tag 固定：

    eg-rsa-v1-freeze

含义：

    eg-rsa-v1-freeze 指向某一个固定 commit。
    后续 eg-rsa-v2-dev 如何修改，都不会改变这个 tag。

查看 tag：

    git tag

查看 tag 指向：

    git show eg-rsa-v1-freeze --no-patch --oneline

回到 V1：

    git checkout eg-rsa-v1-freeze

基于 V1 新建复现分支：

    git checkout -b eg-rsa-v1-reproduce eg-rsa-v1-freeze

### 2.2 V2 开发分支

建议 V2 在独立分支开发：

    eg-rsa-v2-dev

创建方式：

    git checkout eg-rsa-dev
    git pull origin eg-rsa-dev
    git checkout -b eg-rsa-v2-dev
    git push -u origin eg-rsa-v2-dev

后续所有 V2 代码脚本都在：

    eg-rsa-v2-dev

执行。

---

## 3. 当前问题

V1 当前流程是：

    load initial_schema_path
    -> compile reward
    -> train PPO
    -> collect trajectories
    -> attribution
    -> semantic outcome
    -> failure diagnosis
    -> memory retrieval
    -> LLM edit
    -> edit validation
    -> audit
    -> apply edit / rollback

V1 的问题：

    initial_schema 是人工写的
    diagnostics 是人工写的
    高级语义谓词已经被人工给出
    LLM 主要是在已有 schema 上改权重、禁用组件、添加有限 event rule

因此 V1 只能作为：

    controlled mechanism verification

不能作为：

    fully autonomous reward discovery

---

## 4. V2 核心思想

V2 的主线：

    primitive task interface
    -> LLM bootstrap initial schema
    -> validate generated schema
    -> compile reward
    -> RL training
    -> trajectory diagnostics
    -> LLM schema evolution
    -> validate edit
    -> audit / rollback
    -> next schema

V2 不直接给 LLM 以下人工高级谓词：

    landing_region
    safe_contact
    stable_landing_condition
    approach_region_score
    landing_quality
    stability

如果这些概念出现，必须是 LLM 根据 primitive variables 自己生成的，而不是我们提前提供。

---

## 5. 配置设计

### 5.1 不建议使用 v1/v2 字段

不建议：

    eg_rsa:
      version: v2

建议：

    eg_rsa:
      schema_source:
        type: llm_bootstrap

或者：

    eg_rsa:
      schema_source:
        type: manual

这样主框架不是 V1/V2 分支，而是 schema 来源不同。

### 5.2 manual schema source 配置

用于复用旧逻辑：

    eg_rsa:
      schema_source:
        type: manual
        initial_schema_path: configs/eg_rsa_examples/lunar_lander_initial_schema.json

### 5.3 llm bootstrap schema source 配置

用于 V2：

    eg_rsa:
      schema_source:
        type: llm_bootstrap
        primitive_interface_path: configs/eg_rsa_examples/lunar_lander_primitive_interface.json
        output_subdir: bootstrap
        reuse_if_exists: true
        backend: deepseek
        model: deepseek-v4-pro
        credential_env: DEEPSEEK_API_KEY

兼容旧配置：

    如果 eg_rsa.schema_source 不存在，但 eg_rsa.bootstrap.enabled=true，
    则自动视为 llm_bootstrap。

    如果 eg_rsa.schema_source 不存在，且 eg_rsa.bootstrap.enabled 不存在，
    则自动视为 manual。

---

## 6. 新增文件设计

### 6.1 eg_rsa/schema_sources/base.py

作用：

    定义 SchemaSource 抽象类。

建议代码结构：

    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Any, Dict

    class SchemaSource(ABC):
        @abstractmethod
        def load_or_create(self) -> RewardSchema:
            pass

接口：

    load_or_create() -> RewardSchema

职责：

    返回一个可用的 RewardSchema。
    不负责训练。
    不负责编辑。
    不负责 audit。

---

### 6.2 eg_rsa/schema_sources/manual.py

作用：

    旧 V1 逻辑封装。
    从 initial_schema_path 读取 schema。

建议类名：

    ManualSchemaSource

初始化参数：

    config: Dict[str, Any]
    output_dir: Path

方法：

    load_or_create() -> RewardSchema

逻辑：

    path = config["eg_rsa"]["schema_source"]["initial_schema_path"]
    if missing fallback to config["eg_rsa"]["initial_schema_path"]
    return RewardSchema.from_dict(json.load(path))

---

### 6.3 eg_rsa/schema_sources/llm_bootstrap.py

作用：

    V2 bootstrap schema source。
    调用 BootstrapAgent 生成 initial schema。

建议类名：

    LLMBootstrapSchemaSource

初始化参数：

    config: Dict[str, Any]
    output_dir: Path
    llm_client: Any
    task_description_loader: Callable

方法：

    load_or_create() -> RewardSchema

输出目录：

    output_dir/bootstrap/
        bootstrap_prompt.txt
        bootstrap_response.json
        generated_initial_schema.json
        generated_diagnostics.yml
        bootstrap_report.json

逻辑：

    1. 读取 schema_source 配置
    2. 找到 primitive_interface_path
    3. 如果 reuse_if_exists=true 且 generated_initial_schema.json 存在：
           直接读取 generated_initial_schema.json
    4. 否则调用 BootstrapAgent
    5. 保存所有 bootstrap artifacts
    6. 调用 BootstrapSchemaValidator
    7. 返回 RewardSchema

---

### 6.4 eg_rsa/schema_sources/factory.py

作用：

    根据 config 创建 schema source。

建议函数：

    build_schema_source(
        config: Dict[str, Any],
        output_dir: Path,
        llm_client: Any,
        task_description_loader: Callable,
    ) -> SchemaSource

逻辑：

    source_cfg = config["eg_rsa"].get("schema_source")

    if source_cfg exists:
        source_type = source_cfg["type"]
    elif config["eg_rsa"].get("bootstrap", {}).get("enabled", False):
        source_type = "llm_bootstrap"
    else:
        source_type = "manual"

    if source_type == "manual":
        return ManualSchemaSource(...)

    if source_type == "llm_bootstrap":
        return LLMBootstrapSchemaSource(...)

    else:
        raise ValueError

---

### 6.5 eg_rsa/llm/bootstrap_agent.py

作用：

    生成 initial schema 和 diagnostic spec。

建议类名：

    BootstrapAgent

初始化参数：

    llm_client

方法：

    generate_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: str,
    ) -> Dict[str, Any]

返回格式：

    {
      "initial_schema": {...},
      "diagnostics": {...},
      "bootstrap_report": {
        "design_rationale": "...",
        "assumptions": [...],
        "risk_notes": [...],
        "forbidden_terms_checked": true
      }
    }

Prompt 要求：

    1. 只能使用 primitive variables
    2. 不能使用 forbidden terms
    3. 不能使用 official environment reward
    4. 每个 component 必须有 semantic_role
    5. 每个 event predicate 必须可验证
    6. 输出必须是 JSON
    7. 不要输出 markdown

第一版约束：

    为了尽快跑通，可以允许 LLM 生成 formula_component 和 event_predicate，
    但必须经过 FormulaValidator。

---

### 6.6 eg_rsa/reward/formula_validator.py

作用：

    检查 LLM 生成的 formula / condition 是否安全。

建议类名：

    FormulaValidator

核心方法：

    validate_expression(
        expr: str,
        allowed_variables: set[str],
        allowed_functions: set[str],
    ) -> ValidationResult

检查内容：

    1. Python AST 解析成功
    2. 不允许 Import
    3. 不允许 Attribute
    4. 不允许 Subscript
    5. 不允许 Lambda
    6. 不允许 Call 非白名单函数
    7. 不允许变量不在 allowed_variables
    8. 不允许字符串拼接
    9. 不允许访问 __builtins__
    10. 不允许 eval / exec / open / os / sys

允许节点：

    Expression
    BinOp
    UnaryOp
    BoolOp
    Compare
    Name
    Load
    Constant
    Call
    Add
    Sub
    Mult
    Div
    Pow
    Mod
    USub
    And
    Or
    Not
    Lt
    LtE
    Gt
    GtE
    Eq
    NotEq

---

### 6.7 eg_rsa/reward/safe_formula_eval.py

作用：

    安全计算 formula_component。

建议函数：

    safe_eval_formula(
        expr: str,
        variables: Dict[str, float],
        allowed_functions: Dict[str, Callable],
    ) -> float

要求：

    调用前必须通过 FormulaValidator。
    eval 时 __builtins__ 必须为空。
    locals 只包含 variables 和 allowed_functions。

示例：

    value = eval(
        compiled_expr,
        {"__builtins__": {}},
        safe_locals,
    )

---

### 6.8 eg_rsa/reward/bootstrap_schema_validator.py

作用：

    验证 LLM 生成的 initial schema。

建议类名：

    BootstrapSchemaValidator

方法：

    validate_schema(
        schema: Dict[str, Any],
        primitive_interface: Dict[str, Any],
    ) -> ValidationResult

检查内容：

    1. schema 是 dict
    2. schema 有 version
    3. schema 有 components list
    4. schema 有 event_rules list
    5. 每个 component 有 name/type/weight/enabled/semantic_role
    6. semantic_role 属于允许集合
    7. name 不包含 forbidden terms
    8. formula 不使用 forbidden variables
    9. formula 通过 FormulaValidator
    10. event condition 通过 FormulaValidator
    11. 至少有一个 dense_guidance
    12. 至少有一个 terminal_success 或 safety_constraint
    13. 权重是有限数值
    14. 没有 NaN / inf

---

## 7. Reward Schema 新格式

### 7.1 formula_component

示例：

    {
      "name": "r_centering",
      "type": "formula_component",
      "weight": 1.0,
      "formula": "1.0 - min(abs(x), 1.0)",
      "clip": [-1.0, 1.0],
      "enabled": true,
      "semantic_role": "dense_guidance",
      "reward_timing": "dense",
      "behavior_channel": "position"
    }

含义：

    每一步根据公式计算 dense reward。

字段：

    name: str
    type: formula_component
    weight: float
    formula: str
    clip: [low, high] or null
    enabled: bool
    semantic_role: str
    reward_timing: str
    behavior_channel: str

---

### 7.2 conditional_formula_component

示例：

    {
      "name": "r_slow_near_ground",
      "type": "conditional_formula_component",
      "weight": 1.5,
      "condition": "y < 0.5",
      "formula": "1.0 - min(abs(vy), 1.0)",
      "clip": [0.0, 1.0],
      "enabled": true,
      "semantic_role": "stability_quality",
      "reward_timing": "dense",
      "behavior_channel": "descent"
    }

含义：

    condition 为 true 时才计算 formula。

---

### 7.3 event_predicate

示例：

    {
      "name": "r_soft_two_leg_touchdown_once",
      "type": "event_predicate",
      "weight": 80.0,
      "condition": "left_contact and right_contact and abs(vy) < 0.3 and abs(angle) < 0.3",
      "one_time": true,
      "duration_steps": 3,
      "enabled": true,
      "semantic_role": "terminal_success",
      "reward_timing": "sparse_event",
      "behavior_channel": "success"
    }

含义：

    condition 连续满足 duration_steps 后支付 event reward。
    one_time=true 表示每个 episode 只支付一次。

---

## 8. SafeRewardCompiler 修改设计

当前 SafeRewardCompiler 支持已有类型：

    metric_value
    metric_delta
    action_penalty
    event_bonus
    ...

V2 需要新增：

    formula_component
    conditional_formula_component
    event_predicate

### 8.1 formula_component 编译逻辑

伪代码：

    if ctype == "formula_component":
        raw = safe_eval_formula(component["formula"], primitive_vars, allowed_functions)

### 8.2 conditional_formula_component 编译逻辑

伪代码：

    if ctype == "conditional_formula_component":
        cond = safe_eval_formula(component["condition"], primitive_vars, allowed_functions)
        if bool(cond):
            raw = safe_eval_formula(component["formula"], primitive_vars, allowed_functions)
        else:
            raw = 0.0

### 8.3 event_predicate 编译逻辑

伪代码：

    if event_rule["type"] == "event_predicate":
        cond = bool(safe_eval_formula(event_rule["condition"], primitive_vars, allowed_functions))
        update duration counter
        if duration counter >= duration_steps:
            if one_time and already_fired:
                raw = 0.0
            else:
                raw = 1.0
                mark fired

---

## 9. Primitive Variables 映射

V2 formula 需要 primitive variables。

建议新增函数：

    build_primitive_vars(obs_map, action, state_flags) -> Dict[str, float]

输入：

    obs_map
    action
    state_flags

输出：

    {
      "x": ...,
      "y": ...,
      "vx": ...,
      "vy": ...,
      "angle": ...,
      "angular_velocity": ...,
      "left_contact": ...,
      "right_contact": ...,
      "main_engine": ...,
      "side_engine": ...
    }

映射策略：

    优先从 obs_map 中读取同名字段。
    如果 obs_map 没有，则尝试从 task_metrics 或 state_flags 读取。
    action 根据环境动作格式转换。

注意：

    这一步是工程关键。
    如果 env wrapper 没暴露 x/y/vx/vy 等名字，需要在 wrapper 中补充 obs_map 字段。

---

## 10. Runner 修改设计

Runner 当前第一步是：

    schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))

修改为：

    self.schema_source = build_schema_source(...)
    schema = self.schema_source.load_or_create()

Runner 不需要知道 V1/V2。

### 10.1 EGRSARunner.__init__ 增加

    from eg_rsa.schema_sources.factory import build_schema_source

    self.schema_source = build_schema_source(
        config=self.config,
        output_dir=self.output_dir,
        llm_client=llm_client,
        task_description_loader=self._load_task_description,
    )

### 10.2 EGRSARunner.run 修改

旧：

    schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))

新：

    schema = self.schema_source.load_or_create()

其余 loop 不动。

---

## 11. 测试计划

### 11.1 查看 tag

    git tag
    git show eg-rsa-v1-freeze --no-patch --oneline
    git log --oneline --decorate -5

### 11.2 创建 V2 分支

    git checkout eg-rsa-dev
    git pull origin eg-rsa-dev
    git checkout -b eg-rsa-v2-dev
    git push -u origin eg-rsa-v2-dev

### 11.3 V1 兼容性测试

创建临时 config：

    cp configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml /tmp/v1_smoke.yml

把：

    iterations: 10
    total_timesteps: 1000000

改成：

    iterations: 1
    total_timesteps: 1000

运行：

    python train_eg_rsa.py --config /tmp/v1_smoke.yml

成功标准：

    不调用 bootstrap
    能读取 initial_schema_path
    能生成 iteration_000/reward_schema.json
    能开始训练

### 11.4 V2 bootstrap smoke

运行：

    python train_eg_rsa.py --config configs/eg_rsa_deepseek_v2_bootstrap_smoke.yml

成功标准：

    输出 bootstrap/generated_initial_schema.json
    输出 bootstrap/bootstrap_response.json
    输出 iteration_000/reward_schema.json
    输出 iteration_000/compiled_reward.py
    训练不崩

---

## 12. 开发顺序

### Patch A: SchemaSource 架构

新增：

    eg_rsa/schema_sources/base.py
    eg_rsa/schema_sources/manual.py
    eg_rsa/schema_sources/factory.py

修改：

    eg_rsa/runner.py

目标：

    runner 使用 schema_source.load_or_create()
    V1 仍可运行

---

### Patch B: BootstrapAgent

新增：

    eg_rsa/llm/bootstrap_agent.py
    eg_rsa/schema_sources/llm_bootstrap.py
    eg_rsa/reward/bootstrap_schema_validator.py

目标：

    V2 能生成 initial schema
    能保存 bootstrap artifacts
    能加载 generated_initial_schema.json

---

### Patch C: FormulaValidator + SafeFormulaEval

新增：

    eg_rsa/reward/formula_validator.py
    eg_rsa/reward/safe_formula_eval.py

目标：

    能安全验证和计算 formula expression

---

### Patch D: SafeRewardCompiler 支持 V2 类型

修改：

    eg_rsa/reward/safe_compiler.py

新增支持：

    formula_component
    conditional_formula_component
    event_predicate

目标：

    LLM 生成的 formula schema 能被编译执行

---

### Patch E: V2 smoke test

运行：

    python train_eg_rsa.py --config configs/eg_rsa_deepseek_v2_bootstrap_smoke.yml

---

## 13. 不做的事情

V2 第一阶段不做：

    不追求大规模实验结果
    不跑 10x1M
    不写最终实验结论
    不修改 V1 tag
    不把官方环境 reward 作为反馈信号
    不让 LLM 直接执行任意 Python reward code

---

## 14. 最终架构目标

最终 runner 应该保持干净：

    schema = self.schema_source.load_or_create()

    for iteration in range(iterations):
        compile
        train
        diagnose
        retrieve memory
        edit
        validate
        audit
        apply / rollback

版本差异由 schema_source 和 schema 类型表达，不由 runner 中大量 if v1 / if v2 表达。

这保证：

    V1 被 tag 保存
    V2 是主线开发
    主框架不被 V1/V2 逻辑污染

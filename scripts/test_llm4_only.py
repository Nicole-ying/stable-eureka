"""Test LLM #4 only — using existing experiment's bootstrap data."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime
from eg_rsa.llm.deepseek_client import DeepSeekClient
from eg_rsa.single_chain.evidence import EvidenceBuilder
from eg_rsa.single_chain.expert_memory import build_memory_context_md

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"

# Use existing experiment
run = Path("/home/utseus22/stable-eureka-nicole/experiments/eg_rsa_single_chain/lunar_lander_single_chain_3x300k_smoke_20260621_092110")

# ---- 4 inputs ----
# 1. task_model.md
task_model_md = (run / "agents" / "task_model.md").read_text()

# 2. current_reward_code (from bootstrap's reward dir)
current_reward_code = (run / "reward" / "reward_code.py").read_text()

# 3. iteration_diagnostics (from bootstrap training)
diagnostics_md = EvidenceBuilder.build_diagnostics_md(run)
(run / "training" / "iteration_diagnostics.md").write_text(diagnostics_md)

# 4. memory_context (empty for first iteration — no prior transitions)
memory_context_md = build_memory_context_md(
    current_evidence=json.loads((run / "iteration_evidence.json").read_text()),
    retrieved_memory=[],
    strategy_decision={"next_search_hypothesis": "Initial revision from bootstrap"},
)

# ---- Build prompt ----
prompt_tmpl = (PROMPT_DIR / "expert_reward_revision_prompt.txt").read_text()
prompt = (prompt_tmpl
    .replace("{{task_model_md}}", task_model_md)
    .replace("{{current_reward_code}}", current_reward_code)
    .replace("{{iteration_diagnostics}}", diagnostics_md)
    .replace("{{expert_memory_context}}", memory_context_md))

print(f"Prompt: {len(prompt)} chars")
print(f"  task_model: {len(task_model_md)} chars")
print(f"  reward_code: {len(current_reward_code)} chars")
print(f"  diagnostics: {len(diagnostics_md)} chars")
print(f"  memory: {len(memory_context_md)} chars")
print("Calling DeepSeek API...")

# ---- Call LLM ----
client = DeepSeekClient(model="deepseek-chat", credential_env="DEEPSEEK_API_KEY",
                         temperature=0.1, timeout=300)
output = client.generate(prompt)

# ---- Save ----
out_dir = ROOT / "logs" / "llm4_test" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "expert_reward_revision.md").write_text(output)
(out_dir / "prompt.txt").write_text(prompt)

print(f"\nDone. Output: {len(output)} chars")
print("=" * 60)
print(output[:5000])
print(f"\nFull output: {out_dir / 'expert_reward_revision.md'}")

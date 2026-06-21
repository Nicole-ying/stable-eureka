"""Test ExpertRewardDesignerAgent output quality."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from eg_rsa.llm.deepseek_client import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"

# 1. Load prompt template
prompt_tmpl = (PROMPT_DIR / "expert_reward_designer_prompt.txt").read_text(encoding="utf-8")

# 2. Load task_model.md (from previous test)
task_model_md = (ROOT / "logs" / "task_model_test" / "task_model_output.md").read_text(encoding="utf-8")

# 3. Fill template
prompt = prompt_tmpl.replace("{{task_model_md}}", task_model_md)

print(f"Prompt length: {len(prompt)} chars")
print("Calling DeepSeek API...")

# 4. Call LLM
client = DeepSeekClient(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    credential_env="DEEPSEEK_API_KEY",
    temperature=0.1,
    timeout=300,
)

raw = client.generate(prompt)

# 5. Save output
out_dir = ROOT / "logs" / "reward_designer_test"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "expert_reward_design_output.md"
out_path.write_text(raw, encoding="utf-8")

print(f"\nDone. Output saved to: {out_path}")
print(f"Total: {len(raw)} chars")
print("=" * 70)
print(raw[:6000])
print(f"\n... (total {len(raw)} chars)")

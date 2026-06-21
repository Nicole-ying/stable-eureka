"""Test LLM #1 + LLM #2 pipeline — no training."""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime
from eg_rsa.llm.deepseek_client import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"

# --- Config ---
out_dir = ROOT / "logs" / "bootstrap_test" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)
client = DeepSeekClient(model="deepseek-chat", credential_env="DEEPSEEK_API_KEY",
                         temperature=0.1, timeout=300)

# --- LLM #1: TaskModelAgent ---
print("=" * 60)
print("LLM #1: TaskModelAgent")
print("=" * 60)

tm_prompt_tmpl = (PROMPT_DIR / "task_model_prompt.txt").read_text(encoding="utf-8")
task_desc = (ROOT / "envs" / "lunar_lander" / "task_description.txt").read_text(encoding="utf-8")
step_code = (ROOT / "envs" / "lunar_lander" / "step.py").read_text(encoding="utf-8")
tm_prompt = tm_prompt_tmpl.replace("{{task_description}}", task_desc).replace("{{step_code}}", step_code)

print(f"Prompt: {len(tm_prompt)} chars → Calling API...")
task_model_md = client.generate(tm_prompt)
(out_dir / "task_model.md").write_text(task_model_md, encoding="utf-8")
print(f"Output: {len(task_model_md)} chars → {out_dir / 'task_model.md'}")

# --- LLM #2: ExpertRewardDesignerAgent ---
print(f"\n{'=' * 60}")
print("LLM #2: ExpertRewardDesignerAgent")
print("=" * 60)

rd_prompt_tmpl = (PROMPT_DIR / "expert_reward_designer_prompt.txt").read_text(encoding="utf-8")
rd_prompt = rd_prompt_tmpl.replace("{{task_model_md}}", task_model_md)

print(f"Prompt: {len(rd_prompt)} chars → Calling API...")
designer_md = client.generate(rd_prompt)
(out_dir / "expert_reward_design.md").write_text(designer_md, encoding="utf-8")
print(f"Output: {len(designer_md)} chars → {out_dir / 'expert_reward_design.md'}")

# --- Extract and verify code blocks ---
def extract_code_block(md, lang):
    m = re.search(rf"```{lang}\s*\n(.*?)```", md, re.DOTALL)
    return m.group(1).strip() if m else None

py = extract_code_block(designer_md, "python")
js = extract_code_block(designer_md, "json")

print(f"\n{'=' * 60}")
print("Verification")
print("=" * 60)

if py:
    import ast
    try:
        ast.parse(py)
        print(f"✅ reward_code: {len(py)} chars, AST OK")

        # Key checks
        checks = {
            "terminal uses not self.game_over": "not self.game_over" in py,
            "terminal NOT handcrafted legs": "left_leg ==" not in (py.split("terminal")[1].split("\n")[0] if "terminal" in py else ""),
            "has delta form": "max(0, prev" in py or "max(0, self._prev" in py,
            "one-shot guard": "self._" in py and ("True" in py.split("_")[-1] if "_" in py else True),
            "import math": "import math" in py,
            "individual_reward": "individual_reward" in py,
        }
        for label, ok in checks.items():
            print(f"  {'✅' if ok else '❌'} {label}")
    except SyntaxError as e:
        print(f"❌ reward_code SyntaxError: {e}")
else:
    print("❌ No python code block found")

if js:
    try:
        schema = json.loads(js)
        print(f"✅ reward_schema: {len(schema)} components JSON OK")
        for c in schema:
            print(f"  {c.get('id')}: {c.get('category')}")
    except json.JSONDecodeError as e:
        print(f"❌ reward_schema JSON Error: {e}")
else:
    print("❌ No json code block found")

print(f"\nFull outputs: {out_dir}")

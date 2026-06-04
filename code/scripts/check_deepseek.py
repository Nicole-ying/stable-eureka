from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_reward_evolver.llm import DeepSeekClient


def main() -> None:
    client = DeepSeekClient(model="deepseek-chat")
    response = client.complete(
        "Return only this Python code: "
        "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0): "
        "return float(original_reward)"
    )
    print(response[:500])


if __name__ == "__main__":
    main()


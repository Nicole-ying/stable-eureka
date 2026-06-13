import base64
import json
from pathlib import Path

import ollama
from openai import OpenAI

from .config import ModelConfig


class ModelGateway:
    """Thin model abstraction without external agent frameworks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider.lower()
        self.openai_client = OpenAI() if self.provider == "openai" else None
        self.ollama_client = ollama.Client(host=config.ollama_host) if self.provider == "ollama" else None

    def chat(self, system: str, user: str) -> str:
        if self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.config.temperature},
            )
            return response["message"]["content"]

        # mock provider for local pipeline tests
        return (
            "```python\n"
            "def compute_reward(obs, action, next_obs, env_reward, done, info):\n"
            "    reward = float(env_reward)\n"
            "    if done:\n"
            "        reward += 1.0\n"
            "    return reward\n"
            "```\n"
            "RATIONALE: keep environment reward and lightly shape terminal behavior."
        )

    def judge_video(self, system_prompt: str, rubric: str, video_path: Path) -> dict:
        if self.provider == "openai":
            b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
            data_url = f"data:video/mp4;base64,{b64}"
            response = self.openai_client.chat.completions.create(
                model=self.config.vlm_model,
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": rubric},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": f"judge_parse_error: {content[:200]}"}

        if self.provider == "ollama":
            # Many local models lack direct video understanding; use text-side judging fallback.
            response = self.ollama_client.chat(
                model=self.config.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{rubric}\nVideo file path: {video_path}. "
                            "If no visual access, return score 0 and explain limitation."
                        ),
                    },
                ],
                format="json",
                options={"temperature": 0.2},
            )
            try:
                return json.loads(response["message"]["content"])
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": "ollama_json_parse_error"}

        return {"score": 0.0, "reason": "mock_judge_no_vision"}

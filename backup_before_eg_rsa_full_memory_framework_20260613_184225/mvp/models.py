import base64
import json
import os
from pathlib import Path

import ollama
from openai import OpenAI

from .config import ModelConfig


class ModelGateway:
    """Thin model abstraction without external agent frameworks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider.lower()

        self.openai_client = None
        self.ollama_client = None

        if self.provider == "openai":
            kwargs = {
                "api_key": os.environ.get(config.openai_api_key_env),
            }
            if config.openai_base_url:
                kwargs["base_url"] = config.openai_base_url
            self.openai_client = OpenAI(**kwargs)

        elif self.provider == "deepseek":
            api_key = os.environ.get(config.deepseek_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing DeepSeek API key. Please export {config.deepseek_api_key_env}=<your_key>"
                )
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=config.deepseek_base_url,
            )

        elif self.provider == "ollama":
            self.ollama_client = ollama.Client(host=config.ollama_host)

        elif self.provider == "mock":
            pass

        else:
            raise ValueError(
                f"Unsupported model provider: {config.provider}. "
                "Expected one of: openai, deepseek, ollama, mock."
            )

    def _chat_openai_compatible(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.config.llm_model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        if self.provider == "deepseek":
            thinking = getattr(self.config, "deepseek_thinking", "disabled")
            reasoning_effort = getattr(self.config, "deepseek_reasoning_effort", None)

            if thinking:
                kwargs["extra_body"] = {
                    "thinking": {
                        "type": str(thinking),
                    }
                }

            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        response = self.openai_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""

        if not content.strip():
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            raise RuntimeError(
                "LLM returned empty message.content. "
                f"provider={self.provider}, model={self.config.llm_model}, "
                f"finish_reason={finish_reason}, usage={usage}. "
                "For DeepSeek code generation, use deepseek_thinking: disabled and increase max_tokens if needed."
            )

        return content

    def chat(self, system: str, user: str) -> str:
        if self.provider in ("openai", "deepseek"):
            return self._chat_openai_compatible(system, user)

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.config.temperature},
            )
            content = response["message"]["content"] or ""
            if not content.strip():
                raise RuntimeError(
                    f"Ollama returned empty content. model={self.config.llm_model}"
                )
            return content

        # mock provider
        return (
            "```python\n"
            "def compute_reward(obs, action, next_obs, done, info):\n"
            "    obs_arr = np.asarray(obs, dtype=float).reshape(-1)\n"
            "    next_arr = np.asarray(next_obs, dtype=float).reshape(-1)\n"
            "    delta = next_arr - obs_arr\n"
            "    progress = float(np.clip(np.linalg.norm(obs_arr) - np.linalg.norm(next_arr), -5.0, 5.0))\n"
            "    stability = float(-0.05 * np.tanh(np.linalg.norm(delta)))\n"
            "    act_arr = np.asarray(action, dtype=float).reshape(-1)\n"
            "    effort = float(-0.01 * np.tanh(np.linalg.norm(act_arr)))\n"
            "    terminal = float(-1.0 if done else 0.0)\n"
            "    total = progress + stability + effort + terminal\n"
            "    components = {\n"
            "        'progress': progress,\n"
            "        'stability': stability,\n"
            "        'effort': effort,\n"
            "        'terminal': terminal,\n"
            "    }\n"
            "    return float(total), components\n"
            "```\n"
            "RATIONALE: clean bounded transition reward using only public transition inputs."
        )

    def judge_video(self, system_prompt: str, rubric: str, video_path: Path) -> dict:
        if self.provider in ("openai", "deepseek"):
            # DeepSeek 当前主要用于文本 reward generation。
            # 如果没有真实多模态能力，这里返回 0 分，不影响 selection_score；
            # selection_score 仍由 private evaluator return 决定。
            if self.provider == "deepseek":
                return {
                    "score": 0.0,
                    "reason": "deepseek_text_only_judge_skipped",
                    "strengths": [],
                    "weaknesses": [],
                }

            b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
            data_url = f"data:image/gif;base64,{b64}"
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

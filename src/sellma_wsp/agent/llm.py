"""Chat backends for free models: a local Ollama server, or any OpenAI-compatible endpoint.

Only the standard library is used, so no API SDK is required.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class LLMConfig:
    backend: str = "ollama"            
    model: str = "qwen3:4b"
    base_url: str = ""                 
    api_key_env: str = "OPENAI_API_KEY"  
    temperature: float = 0.6
    seed: int = 0
    max_tokens: int = 4096
    num_ctx: int = 12288               
    think: bool = False                
    timeout_s: int = 900


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class ChatBackend:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.last_thinking = "" 

    def chat(self, messages: list[dict]) -> str:
        delay = 5
        for attempt in range(6):
            try:
                return _THINK.sub("", self._chat(messages)).strip()
            except urllib.error.HTTPError as e:  
                if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise RuntimeError(f"LLM HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 5:
                    raise RuntimeError(f"LLM unreachable: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise AssertionError("unreachable")

    def _chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {k: v for k, v in asdict(self.cfg).items()}


class OllamaBackend(ChatBackend):
    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        self.url = (cfg.base_url or "http://localhost:11434").rstrip("/")

    def _chat(self, messages):
        out = _post(f"{self.url}/api/chat", {
            "model": self.cfg.model, "messages": messages, "stream": False, "think": self.cfg.think,
            "options": {"temperature": self.cfg.temperature, "seed": self.cfg.seed,
                        "num_ctx": self.cfg.num_ctx, "num_predict": self.cfg.max_tokens},
        }, {}, self.cfg.timeout_s)
        self.last_thinking = out["message"].get("thinking") or ""
        return out["message"]["content"]

    def describe(self):
        d = super().describe()
        try: 
            tags = _get(f"{self.url}/api/tags")["models"]
            d["model_digest"] = next((m["digest"] for m in tags if m["name"] == self.cfg.model), None)
            d["ollama_version"] = _get(f"{self.url}/api/version").get("version")
        except Exception as e:  
            d["model_digest_error"] = str(e)
        return d


class OpenAICompatBackend(ChatBackend):
    """OpenRouter (`:free` models), Groq, Gemini's OpenAI endpoint, LM Studio, vLLM, …"""

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        if not cfg.base_url:
            raise ValueError("--base-url is required for the openai backend")
        self.url = cfg.base_url.rstrip("/")
        self.key = os.environ.get(cfg.api_key_env, "")

    def _chat(self, messages):
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        out = _post(f"{self.url}/chat/completions", {
            "model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
            "seed": self.cfg.seed, "max_tokens": self.cfg.max_tokens,
        }, headers, self.cfg.timeout_s)
        msg = out["choices"][0]["message"]
        self.last_thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return msg["content"] or ""


class ScriptedBackend(ChatBackend):
    """Replays fixed replies; used for tests and harness dry runs."""

    def __init__(self, cfg: LLMConfig, replies: list[str]):
        super().__init__(cfg)
        self.replies = list(replies)

    def _chat(self, messages):
        return self.replies.pop(0) if self.replies else "Done."


def make_backend(cfg: LLMConfig) -> ChatBackend:
    if cfg.backend == "ollama":
        return OllamaBackend(cfg)
    if cfg.backend == "openai":
        return OpenAICompatBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")

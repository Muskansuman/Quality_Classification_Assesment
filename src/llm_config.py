"""Resolve LLM provider settings (Ollama open-source vs OpenAI cloud)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


@dataclass(frozen=True)
class LlmSettings:
    provider: str  # ollama | openai
    model: str
    api_key: str
    base_url: str | None  # set for Ollama / compatible gateways
    engine: str


def resolve_llm_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LlmSettings | None:
    """Return settings when an LLM backend is configured; None → use heuristic."""
    prov = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if prov == "ollama":
        return LlmSettings(
            provider="ollama",
            model=model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            engine="ollama",
        )

    if prov == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key or key.startswith("sk-your"):
            return None
        return LlmSettings(
            provider="openai",
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            engine="openai",
        )

    return None

"""
Be able to switch LLMs without touching any other code.
"""

import os
import httpx
from abc import ABC, abstractmethod
from anthropic import AsyncAnthropic


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """
        Send a prompt and return the response text.
        """


# ----- Claude -----
class ClaudeProvider(LLMProvider):
    async def complete(self, system: str, user: str) -> str:
        client = AsyncAnthropic()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ----- Ollama -----
class OllamaProvider(LLMProvider):
    async def complete(self, system: str, user: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2",  # or whichever model you have pulled
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]


# ----- Factory -----
def get_llm_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama")  # default to free
    if provider == "claude":
        return ClaudeProvider()
    elif provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider: {provider}")

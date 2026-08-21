"""
BaseLLM — the contract every language model provider must follow.

Same pattern as BaseEmbedder and BaseLoader: today it's Groq (Llama 3.3).
Later we can add Azure OpenAI or Anthropic behind this same contract. The
RAG chain that calls this never needs to know or care which provider is
answering -- it just calls `.generate()`.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a prompt to the LLM and return its text response."""
        raise NotImplementedError
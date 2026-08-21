"""
GroqLLM — calls Llama models hosted on Groq's fast inference API.

Design note on fallback:
    LLM API calls can fail -- timeout, rate limit, provider outage. A
    production RAG system should never let this crash the whole request; it
    should return a clear, honest message instead. We'll make this more
    sophisticated in Step 8 (retries, backoff); for now, catch and return a
    safe message so the caller always gets *something* usable back.
"""

from groq import Groq

from app.config import settings
from app.generation.base import BaseLLM
from app.logger import get_logger

log = get_logger(__name__)


class GroqLLM(BaseLLM):
    def __init__(self):
        if not settings.groq_api_key:
            log.error("groq_api_key_missing")
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            answer = response.choices[0].message.content
            log.info("llm_generation_succeeded", model=self.model, chars=len(answer))
            return answer

        except Exception as e:
            log.error("llm_generation_failed", model=self.model, error=str(e))
            return (
                "I'm having trouble generating an answer right now due to a "
                "system issue. Please try again in a moment."
            )
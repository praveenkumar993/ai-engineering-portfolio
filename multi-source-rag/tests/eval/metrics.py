"""
Metrics — simple, explainable scoring functions for eval.

Why keyword-based checks instead of an LLM judging the answer:
    "LLM-as-judge" adds cost, latency, and a new source of unreliability --
    the judge model can be wrong too. Keyword presence is blunt, but it's
    deterministic, free, instant, and fully explainable: you can look at
    any failure and know EXACTLY why it failed.
"""

from app.generation.response import RAGResponse


def check_answer_correctness(response: RAGResponse, expected_keywords: list[str]) -> bool:
    answer_lower = response.answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def check_retrieval_source(response: RAGResponse, expected_source_contains: str | None) -> bool:
    if expected_source_contains is None:
        return len(response.sources) == 0

    return any(
        expected_source_contains in chunk.source_path for chunk in response.sources
    )
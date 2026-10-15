"""
RAGPipeline — orchestrates retrieve -> rerank -> generate, with an explicit
fallback check at each stage instead of assuming every stage succeeds.

The fallback chain:
    1. Hybrid retrieval returns candidates -> if empty, stop and say so.
    2. Reranker scores them -> if the BEST score is below the configured
       threshold, stop: nothing here is relevant enough to answer from.
    3. LLM generates an answer -> if the provider failed, mark LLM_ERROR so
       callers can tell "no context" apart from "LLM broke."

Why the threshold is configurable, not hardcoded:
    Cross-encoder scores are a relative ranking signal, not a calibrated
    probability, so there's no universally correct cutoff. It depends on
    the model and the documents. We tune it against the golden dataset
    (tests/eval) and keep it in config so it can change without a deploy.
"""

from app.config import settings
from app.generation.base import BaseLLM
from app.generation.prompt import build_rag_prompt
from app.generation.response import RAGResponse, ResponseStatus
from app.logger import get_logger
from app.reranker.base import BaseReranker
from app.retrieval.hybrid_retriever import HybridRetriever

log = get_logger(__name__)

LLM_FAILURE_MARKER = "I'm having trouble generating an answer right now"


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        llm: BaseLLM,
        search_top_k: int = 10,
        final_top_k: int = 3,
        min_relevance_score: float | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.search_top_k = search_top_k
        self.final_top_k = final_top_k
        self.min_relevance_score = (
            settings.min_relevance_score
            if min_relevance_score is None
            else min_relevance_score
        )

    def ask(self, question: str) -> RAGResponse:
        # --- Stage 1: hybrid retrieve (semantic + BM25, fused) ---
        candidates = self.retriever.search(question, top_k=self.search_top_k)

        if not candidates:
            log.warning("no_candidates_found", question=question)
            return RAGResponse(
                status=ResponseStatus.NO_CONTEXT,
                answer="I don't have any indexed documents relevant to that question.",
            )

        # --- Stage 2: rerank, then check the best match is relevant enough ---
        scored = self.reranker.rerank_with_scores(
            question, candidates, top_k=self.final_top_k
        )

        if not scored:
            log.warning("reranking_produced_no_results", question=question)
            return RAGResponse(
                status=ResponseStatus.NO_CONTEXT,
                answer="I couldn't find relevant information to answer that question.",
            )

        top_score = scored[0][1]
        if top_score < self.min_relevance_score:
            log.warning(
                "reranking_below_relevance_threshold",
                question=question,
                top_score=top_score,
                threshold=self.min_relevance_score,
            )
            return RAGResponse(
                status=ResponseStatus.NO_CONTEXT,
                answer="I don't have enough information in the provided documents to answer that.",
                top_score=top_score,
            )

        top_chunks = [chunk for chunk, _ in scored]

        # --- Stage 3: generate ---
        prompt = build_rag_prompt(question, top_chunks)
        answer = self.llm.generate(prompt)

        if LLM_FAILURE_MARKER in answer:
            log.error("pipeline_llm_stage_failed", question=question)
            return RAGResponse(
                status=ResponseStatus.LLM_ERROR,
                answer=answer,
                sources=top_chunks,
                top_score=top_score,
            )

        log.info("pipeline_answered_successfully", question=question, sources=len(top_chunks))
        return RAGResponse(
            status=ResponseStatus.OK,
            answer=answer,
            sources=top_chunks,
            top_score=top_score,
        )
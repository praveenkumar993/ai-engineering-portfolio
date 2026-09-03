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
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.search_top_k = search_top_k
        self.final_top_k = final_top_k

    def ask(self, question: str) -> RAGResponse:
        # --- Stage 1: hybrid retrieve (semantic + BM25, fused) ---
        candidates = self.retriever.search(question, top_k=self.search_top_k)

        if not candidates:
            log.warning("no_candidates_found", question=question)
            return RAGResponse(
                status=ResponseStatus.NO_CONTEXT,
                answer="I don't have any indexed documents relevant to that question.",
                sources=[],
            )

        top_chunks = self.reranker.rerank(question, candidates, top_k=self.final_top_k)

        if not top_chunks:
            log.warning("reranking_produced_no_results", question=question)
            return RAGResponse(
                status=ResponseStatus.NO_CONTEXT,
                answer="I couldn't find relevant information to answer that question.",
                sources=[],
            )

        prompt = build_rag_prompt(question, top_chunks)
        answer = self.llm.generate(prompt)

        if LLM_FAILURE_MARKER in answer:
            log.error("pipeline_llm_stage_failed", question=question)
            return RAGResponse(
                status=ResponseStatus.LLM_ERROR,
                answer=answer,
                sources=top_chunks,
            )

        log.info("pipeline_answered_successfully", question=question, sources=len(top_chunks))
        return RAGResponse(
            status=ResponseStatus.OK,
            answer=answer,
            sources=top_chunks,
        )
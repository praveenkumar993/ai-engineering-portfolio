"""
RAGPipeline — orchestrates retrieve -> rerank -> generate, with an explicit
fallback check at each stage instead of assuming every stage succeeds.

Why pull this out of main.py:
    main.py was starting to mix "wiring things together" with "business
    logic" (what to do when retrieval is empty, etc). Separating them means
    main.py just builds the components and calls pipeline.ask(question) --
    and this class becomes reusable later by Step 9's FastAPI endpoint
    without copy-pasting logic.

The fallback chain, explicitly:
    1. Vector search returns candidates -> if empty, stop and say so.
    2. Reranker narrows to top_k -> if that's somehow empty, stop and say so.
    3. LLM generates an answer -> GroqLLM already catches provider failures
       internally and returns a safe string; here we just detect that case
       by checking for its known failure message and mark the response
       status accordingly, so callers can distinguish "no context" from
       "LLM broke."
"""

from app.embeddings.base import BaseEmbedder
from app.generation.base import BaseLLM
from app.generation.prompt import build_rag_prompt
from app.generation.response import RAGResponse, ResponseStatus
from app.logger import get_logger
from app.reranker.base import BaseReranker
from app.vectorstore.base import BaseVectorStore

log = get_logger(__name__)

LLM_FAILURE_MARKER = "I'm having trouble generating an answer right now"


class RAGPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: BaseReranker,
        llm: BaseLLM,
        search_top_k: int = 10,
        final_top_k: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.search_top_k = search_top_k
        self.final_top_k = final_top_k

    def ask(self, question: str) -> RAGResponse:
        query_vector = self.embedder.embed([question])[0]
        candidates = self.vector_store.search(query_vector, top_k=self.search_top_k)

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
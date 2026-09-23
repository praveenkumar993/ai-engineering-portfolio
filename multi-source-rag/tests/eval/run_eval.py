"""
run_eval.py — runs the golden dataset through the real RAG pipeline and
reports pass/fail for each question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.generation.groq_llm import GroqLLM
from app.generation.pipeline import RAGPipeline

from tests.eval.golden_dataset import GOLDEN_DATASET
from tests.eval.metrics import check_answer_correctness, check_retrieval_source

configure_logging()
log = get_logger(__name__)


def build_pipeline_for_eval() -> RAGPipeline:
    validate_startup_config()

    loader = LocalLoader(folder_path="docs")
    documents = loader.load()
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=80)

    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = QdrantStore(collection_name="rag_chunks_eval", dimension=embedder.dimension, url=settings.qdrant_url)
    store.add_chunks(chunks, vectors)

    bm25 = BM25Retriever(chunks)
    retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25_retriever=bm25)

    reranker = CrossEncoderReranker()
    llm = GroqLLM()

    return RAGPipeline(retriever=retriever, reranker=reranker, llm=llm)


def run_eval() -> None:
    try:
        pipeline = build_pipeline_for_eval()
    except ConfigError as e:
        print(f"Cannot run eval -- startup config invalid:\n{e}")
        return

    results = []
    for case in GOLDEN_DATASET:
        response = pipeline.ask(case["question"])

        answer_correct = check_answer_correctness(response, case["expected_keywords"])
        retrieval_correct = check_retrieval_source(response, case["expected_source_contains"])
        passed = answer_correct and retrieval_correct

        results.append({
            "question": case["question"],
            "passed": passed,
            "answer_correct": answer_correct,
            "retrieval_correct": retrieval_correct,
            "answer": response.answer,
        })

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    print(f"\n{'='*60}")
    print(f"EVAL RESULTS: {passed_count}/{total} passed")
    print(f"{'='*60}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['question']}")
        if not r["passed"]:
            print(f"       answer_correct={r['answer_correct']}, retrieval_correct={r['retrieval_correct']}")
            print(f"       got: {r['answer'][:150]}")
        print()

    log.info("eval_completed", total=total, passed=passed_count, pass_rate=round(passed_count/total, 2))


if __name__ == "__main__":
    run_eval()
"""
Golden dataset — a small, hand-written set of questions with known correct
answers, used to measure whether the RAG pipeline is actually working well.

Why hand-write these instead of generating them automatically:
    The whole point of a golden dataset is that WE know the correct answer
    independently of the system we're testing. Since our test document
    (sample_rag_test.pdf) is short and we wrote its content ourselves, we
    can hand-write accurate expected answers with full confidence.

    In a real production system, this dataset would be built by subject
    matter experts reviewing real documents -- same idea, company scale.
"""

GOLDEN_DATASET = [
    {
        "question": "What are the three main stages of a RAG pipeline?",
        "expected_keywords": ["ingestion", "retrieval", "generation"],
        "expected_source_contains": "sample_rag_test.pdf",
    },
    {
        "question": "What is RAG?",
        "expected_keywords": ["retrieval", "language model"],
        "expected_source_contains": "sample_rag_test.pdf",
    },
    {
        "question": "What does reranking do in a production RAG system?",
        "expected_keywords": ["rerank", "relevance"],
        "expected_source_contains": "sample_rag_test.pdf",
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": ["don't have enough information"],
        "expected_source_contains": None,
    },
]
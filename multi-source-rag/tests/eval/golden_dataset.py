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
    # --- Borderline positives: answerable, but phrased far from the source
    # text. These scored -1.7 to -4.3 in the reranking comparison, so they
    # test that the threshold isn't too strict.
    {
        "question": "How do production systems handle failures?",
        "expected_keywords": ["fallback"],
        "expected_source_contains": "sample_rag_test.pdf",
    },
    {
        "question": "What happens to a document before it is stored?",
        "expected_keywords": ["chunk"],
        "expected_source_contains": "sample_rag_test.pdf",
    },
    # --- Near-miss negative: on the same general subject (AI) but NOT in
    # our documents. This tests the opposite side: that the threshold isn't
    # too lenient. It is the hardest refusal case in the set.
    {
        "question": "How many parameters does GPT-4 have?",
        "expected_keywords": ["don't have enough information"],
        "expected_source_contains": None,
    },
]
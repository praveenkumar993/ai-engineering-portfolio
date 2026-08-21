"""
Prompt builder — this is where "grounding" actually happens.

Anti-hallucination isn't a magic setting you flip on. It's mostly just
disciplined prompt engineering: explicitly telling the model what context
it's allowed to use, and explicitly telling it what to do when the answer
isn't in that context (say so, don't guess).
"""

from app.ingestion.models import Chunk

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- Only use information from the context to answer.
- If the answer is not present in the context, say "I don't have enough information in the provided documents to answer that."
- Do not make up facts that are not in the context.
- Keep your answer concise and directly address the question.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    context_blocks = [
        f"[Source {i+1}: {c.source_path}]\n{c.text}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n".join(context_blocks)

    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
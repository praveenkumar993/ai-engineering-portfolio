"""
LLM factory -- one place that builds the chat model every node uses.

Why a LangChain chat model (ChatGroq) instead of our own wrapper?
    LangGraph's prebuilt agents (create_react_agent, Step 7) call
    bind_tools() on the model so it can do tool calling. That method
    exists on LangChain chat models, so we use one here. Every node in
    the project gets its model from this function, so swapping Groq
    for another provider later is a one-line change in this file.

Why temperature defaults to 0:
    Code review should be consistent and grounded, not creative.
    Same diff in, same analysis out, as far as possible.
"""

from langchain_groq import ChatGroq

from app.config import settings


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )
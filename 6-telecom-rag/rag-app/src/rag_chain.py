"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from importlib import import_module
from typing import Any, Sequence

try:
    ChatPromptTemplate = import_module("langchain_core.prompts").ChatPromptTemplate
except ModuleNotFoundError:
    ChatPromptTemplate = import_module("langchain.prompts").ChatPromptTemplate

try:
    StrOutputParser = import_module("langchain_core.output_parsers").StrOutputParser
except ModuleNotFoundError:
    StrOutputParser = import_module("langchain.output_parsers").StrOutputParser

try:
    RunnablePassthrough = import_module("langchain_core.runnables").RunnablePassthrough
except ModuleNotFoundError:
    RunnablePassthrough = import_module("langchain.runnables").RunnablePassthrough

try:
    Document = import_module("langchain_core.documents").Document
except ModuleNotFoundError:
    Document = import_module("langchain.schema").Document

try:
    ChatGroq = import_module("langchain_groq").ChatGroq
except ModuleNotFoundError:
    ChatGroq = import_module("langchain_groq").ChatGroq

try:
    build_retriever = import_module("retriever").build_retriever
except (ImportError, ModuleNotFoundError):
    build_retriever = import_module("retriever").build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: Sequence[Any]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
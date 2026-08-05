"""lesson23 — Cited RAG answers + production safety limits.

Docs reference:
  https://docs.langchain.com/oss/python/langchain/agents
  https://docs.langchain.com/oss/python/langchain/middleware

Builds on lesson21/22 with two production concerns:

1) Structured citations
   - response_format forces topic/summary/sources shape
   - Real-world: UI can render source chips / audit trail

2) Middleware guardrails
   - ToolCallLimitMiddleware: stop runaway tool loops
   - ModelCallLimitMiddleware: cap LLM spend per request

Real-world scenario:
  Enterprise support assistant where every answer must show
  which policy docs were used, and cost/latency must be bounded
  so a confused model cannot call tools forever.

Run:
  ollama serve
  python lesson23.py
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Real-world: curated policy corpus with stable source IDs for citations.
# Compliance teams often require answers to link back to approved docs only.
# ---------------------------------------------------------------------------
RAW_DOCS = [
    Document(
        page_content=(
            "Free plan: reading, writing, listening, and speaking practice "
            "with no credit card required."
        ),
        metadata={"source": "pricing.md", "doc_id": "POL-PRICE-01"},
    ),
    Document(
        page_content=(
            "Premium costs $12/month and includes advanced AI feedback and "
            "personalized study plans."
        ),
        metadata={"source": "pricing.md", "doc_id": "POL-PRICE-02"},
    ),
    Document(
        page_content=(
            "Refunds may be requested within 30 days. Approved refunds are "
            "processed within 7 business days."
        ),
        metadata={"source": "refunds.md", "doc_id": "POL-REF-01"},
    ),
    Document(
        page_content=(
            "Password reset is under Account Settings → Security → Reset Password."
        ),
        metadata={"source": "account.md", "doc_id": "POL-ACC-01"},
    ),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chunks = RecursiveCharacterTextSplitter(
    chunk_size=260, chunk_overlap=40
).split_documents(RAW_DOCS)
retriever = Chroma.from_documents(
    documents=chunks, embedding=embeddings
).as_retriever(search_kwargs={"k": 3})


class CitedAnswer(BaseModel):
    """Structured support answer for UI + audit logs."""

    # Real-world: frontend renders summary text + clickable source chips.
    topic: str = Field(description="Short topic label")
    summary: str = Field(description="Customer-facing answer")
    source_ids: list[str] = Field(
        description="Policy doc IDs used, e.g. POL-REF-01"
    )
    confidence: str = Field(
        description="high | medium | low based on retrieved evidence"
    )


@tool
def search_policy_docs(query: str) -> str:
    """Search approved Cosmic Learning policy documents and return cited snippets."""
    # Real-world: only search the approved/published index, never draft docs.
    docs = retriever.invoke(query)
    if not docs:
        return "No approved policy documents matched."

    lines = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", "UNKNOWN")
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{doc_id} | {source}] {doc.page_content}")
    return "\n\n".join(lines)


agent = create_agent(
    model="ollama:llama3.2",
    tools=[search_policy_docs],
    system_prompt=(
        "You are Cosmic Learning's cited policy assistant.\n"
        "Always call search_policy_docs before answering policy questions.\n"
        "Put used document IDs into source_ids.\n"
        "If evidence is weak, set confidence to low and say what is missing."
    ),
    response_format=CitedAnswer,
    middleware=[
        # Real-world: prevents infinite retrieve loops on ambiguous questions.
        ToolCallLimitMiddleware(run_limit=4),
        # Real-world: hard budget on LLM calls per user request / ticket.
        ModelCallLimitMiddleware(run_limit=6),
    ],
)


if __name__ == "__main__":
    questions = [
        "How long do refunds take?",
        "What does Premium include and how much is it?",
        "How do I reset my password?",
    ]

    for question in questions:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        cited = result.get("structured_response")

        print("=" * 60)
        print("Q:", question)
        if cited is not None:
            print("Topic:", cited.topic)
            print("Summary:", cited.summary)
            print("Sources:", cited.source_ids)
            print("Confidence:", cited.confidence)
        else:
            # Fallback if the local model skips structured output
            print("Raw:", result["messages"][-1].content)
        print()

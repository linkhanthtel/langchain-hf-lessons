"""lesson22 — Multi-tool support agent (RAG + business systems).

Docs reference:
  https://docs.langchain.com/oss/python/langchain/agents
  https://docs.langchain.com/oss/python/langchain/tools

Pattern:
  One agent, many tools. The model chooses which systems to call.

  Customer message
       │
       ▼
  create_agent
       ├─ search_help_docs   (vector RAG over policies)
       ├─ get_order_status   (order service / Postgres)
       └─ estimate_refund    (billing rules engine)

Real-world scenario:
  Cosmic Learning Tier-1 support bot in Intercom/Zendesk.
  Agents must combine:
    - Knowledge base answers (RAG)
    - Live order data (API/DB)  ← never invent order status
    - Simple billing calculations

This is more realistic than pure FAQ RAG, because customers mix
policy questions with account-specific questions in one message.

Run:
  ollama serve
  python lesson22.py
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Real-world: ORDER_DB is a Postgres/Stripe lookup in production.
# Never let the LLM invent order rows — always fetch via a tool.
# ---------------------------------------------------------------------------
ORDER_DB = {
    "ORD-1001": {
        "status": "refund_processing",
        "amount": 89.50,
        "plan": "premium",
        "purchased_days_ago": 12,
    },
    "ORD-1002": {
        "status": "active",
        "amount": 12.00,
        "plan": "premium",
        "purchased_days_ago": 3,
    },
    "ORD-1003": {
        "status": "cancelled",
        "amount": 12.00,
        "plan": "premium",
        "purchased_days_ago": 40,
    },
}

POLICY_DOCS = [
    Document(
        page_content=(
            "Refund policy: customers may request a refund within 30 days "
            "of purchase. Approved refunds are paid within 7 business days."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content=(
            "If a refund is already processing, tell the customer to wait "
            "for the existing refund instead of opening a duplicate request."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content=(
            "Premium plan is $12/month. Free plan has no monthly fee."
        ),
        metadata={"source": "pricing.md"},
    ),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chunks = RecursiveCharacterTextSplitter(
    chunk_size=260, chunk_overlap=40
).split_documents(POLICY_DOCS)
# Real-world: persistent vector index shared by support + website chatbot.
retriever = Chroma.from_documents(
    documents=chunks, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})


@tool
def search_help_docs(query: str) -> str:
    """Search Cosmic Learning help center policies (refunds, pricing, account)."""
    # Real-world: add metadata filters by locale / product / plan tier.
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) or "No docs found."


@tool
def get_order_status(order_id: str) -> str:
    """Look up a customer order by ID like ORD-1001.

    Use this for account-specific status. Do not guess order details.
    """
    # Real-world: SELECT ... FROM orders WHERE id = %s (parameterized).
    order = ORDER_DB.get(order_id.strip().upper())
    if not order:
        return f"No order found for {order_id}."
    return (
        f"order_id={order_id.upper()} status={order['status']} "
        f"amount={order['amount']} plan={order['plan']} "
        f"purchased_days_ago={order['purchased_days_ago']}"
    )


@tool
def estimate_refund(order_id: str) -> str:
    """Estimate whether an order is refund-eligible and the refund amount.

    Combines order data with the 30-day policy window.
    """
    # Real-world: billing rules service / Stripe refund preview API.
    order = ORDER_DB.get(order_id.strip().upper())
    if not order:
        return f"No order found for {order_id}."

    if order["purchased_days_ago"] > 30:
        return (
            f"{order_id.upper()} is outside the 30-day window. "
            "Estimated refund: $0.00"
        )
    if order["status"] == "refund_processing":
        return (
            f"{order_id.upper()} already has a refund processing. "
            f"Do not create another refund for ${order['amount']:.2f}."
        )
    return (
        f"{order_id.upper()} appears eligible. "
        f"Estimated refund: ${order['amount']:.2f}"
    )


agent = create_agent(
    model="ollama:llama3.2",
    tools=[search_help_docs, get_order_status, estimate_refund],
    system_prompt=(
        "You are a Cosmic Learning Tier-1 support agent.\n"
        "Use only tools needed for the CURRENT user message.\n"
        "Workflow:\n"
        "1. Order ID present → call get_order_status.\n"
        "2. Refund eligibility question → call estimate_refund.\n"
        "3. Policy wording → call search_help_docs.\n"
        "4. Pricing-only questions → search_help_docs only.\n"
        "5. Combine tool results into one clear reply.\n"
        "6. Never invent order data or policy text."
    ),
)


if __name__ == "__main__":
    # Real-world tickets often mix policy + account details in one message.
    tickets = [
        "What is your refund policy?",
        "My order is ORD-1001. Can I get a refund?",
        "Check ORD-1003 — I want my money back.",
        "How much is Premium each month?",
    ]

    for ticket in tickets:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": ticket}]}
        )
        print("=" * 60)
        print("TICKET:", ticket)
        print("REPLY:", result["messages"][-1].content)
        print()

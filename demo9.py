"""demo9 — Order-Aware Support Agent API (tools + RAG)

Real-world use:
  Customer portal chat: "Where is ORD-1001?" + "What's your refund policy?"
  The bot must look up LIVE order data (never invent it) AND read policy docs.

What you practice:
  - create_agent with multiple tools (lesson33/34/22)
  - RAG search tool for policies
  - mock order DB tool (replace with Postgres/Stripe in production)
  - FastAPI wrapper so a real frontend can call it

Run:
  ollama serve
  python demo9.py

Open: http://127.0.0.1:8002
"""

from __future__ import annotations

from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Fake order database
# Real world: SELECT from orders WHERE id=%s AND customer_id=%s
# Never let the LLM invent order rows — always fetch via a tool.
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1001": {
        "status": "shipped",
        "item": "Premium annual plan",
        "amount": 99.00,
        "eta_days": 2,
    },
    "ORD-1002": {
        "status": "refund_processing",
        "item": "Premium monthly",
        "amount": 12.00,
        "eta_days": 5,
    },
    "ORD-1003": {
        "status": "active",
        "item": "Premium monthly",
        "amount": 12.00,
        "eta_days": 0,
    },
}

POLICY_DOCS = [
    Document(
        page_content=(
            "Refund policy: request within 30 days of purchase. "
            "Approved refunds are paid within 7 business days."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content=(
            "If refund_processing is already active, do not open a duplicate refund."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content="Premium monthly is $12. Premium annual is $99.",
        metadata={"source": "pricing.md"},
    ),
]

print("demo9: building policy index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
retriever = Chroma.from_documents(
    POLICY_DOCS, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})


@tool
def get_order_status(order_id: str) -> str:
    """Look up a customer order by ID, e.g. ORD-1001.

    Use for account-specific status. Do not guess.
    """
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order found for {order_id}."
    return (
        f"{order_id.upper()}: status={order['status']}, "
        f"item={order['item']}, amount=${order['amount']:.2f}, "
        f"eta_days={order['eta_days']}"
    )


@tool
def search_policies(query: str) -> str:
    """Search Cosmic Learning refund/pricing policies."""
    docs = retriever.invoke(query)
    if not docs:
        return "No policy docs found."
    return "\n\n".join(
        f"[{d.metadata.get('source', 'policy')}] {d.page_content}" for d in docs
    )


# create_agent = model + tools + automatic tool loop (lesson33)
agent = create_agent(
    model="ollama:llama3.2",
    tools=[get_order_status, search_policies],
    system_prompt=(
        "You are Cosmic Learning customer-portal support.\n"
        "Rules:\n"
        "1) If user mentions ORD-xxxx, call get_order_status.\n"
        "2) For refund/pricing policy questions, call search_policies.\n"
        "3) Never invent order data.\n"
        "4) Keep answers short and actionable."
    ),
)


class AskRequest(BaseModel):
    message: str = Field(min_length=1, description="Customer question")
    # Real world: pass authenticated user id and enforce order ownership
    customer_id: str = "demo-user"


class AskResponse(BaseModel):
    request_id: str
    reply: str


app = FastAPI(title="Order-Aware Support Agent", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>demo9 — Order Support Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; }
    textarea { width: 100%; min-height: 90px; padding: 10px; box-sizing: border-box; }
    button { margin-top: 8px; padding: 10px 14px; }
    pre { background: #f6f8fa; padding: 12px; border-radius: 8px; white-space: pre-wrap; }
    .hint { color: #555; }
  </style>
</head>
<body>
  <h1>demo9 — Order-Aware Support Agent</h1>
  <p class="hint">Try: "Where is ORD-1001?" or "Can I refund ORD-1002?" or "How much is Premium monthly?"</p>
  <textarea id="msg">Where is ORD-1001 and how long do refunds take?</textarea>
  <br/>
  <button onclick="ask()">Ask Agent</button>
  <pre id="out">Reply will appear here...</pre>
<script>
async function ask() {
  const message = document.getElementById('msg').value;
  document.getElementById('out').textContent = 'Thinking...';
  const res = await fetch('/agent/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message, customer_id: 'demo-user' })
  });
  const data = await res.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    """Real world: load balancer / uptime checks hit this."""
    return {"status": "ok", "orders_loaded": len(ORDERS)}


@app.get("/orders/demo")
def list_demo_orders():
    """Helper for testers — shows which mock orders exist."""
    return ORDERS


@app.post("/agent/ask", response_model=AskResponse)
def agent_ask(req: AskRequest):
    """Main product endpoint: one customer message → agent reply."""
    # Real world:
    # - authenticate JWT
    # - rate-limit per customer
    # - log request_id for support debugging
    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]}
    )
    reply = result["messages"][-1].content
    return AskResponse(request_id=str(uuid4())[:8], reply=reply)


if __name__ == "__main__":
    print("\nOpen: http://127.0.0.1:8002\n")
    print("Demo orders:", ", ".join(ORDERS.keys()))
    uvicorn.run(app, host="127.0.0.1", port=8002)

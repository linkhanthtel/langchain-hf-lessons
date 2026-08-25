"""demo7 — Cosmic Learning Support Desk (real working mini-app)

What you get:
  1) Web UI at http://127.0.0.1:8000
  2) FAQ chat powered by RAG (like lesson38)
  3) Ticket intake: classify + draft reply (like lesson36 + demo6 ideas)
  4) Optional human approve/edit for high-urgency tickets (like demo6)

This is closer to a real product than demo5/demo6 scripts.

Run:
  ollama serve
  python demo7.py

Then open: http://127.0.0.1:8000
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge base (real apps: Notion / Zendesk / PDFs)
# ---------------------------------------------------------------------------
FAQ_DOCS = [
    Document(
        page_content="Free plan: practice reading, writing, listening, speaking. No credit card.",
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content="Premium costs $12/month and includes advanced AI feedback and study plans.",
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content="Cancel anytime in Billing Settings before the next renewal date.",
        metadata={"source": "billing.md"},
    ),
    Document(
        page_content="Refunds: request within 30 days of purchase. Paid within 7 business days.",
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content="Password reset: Account Settings → Security → Reset Password.",
        metadata={"source": "account.md"},
    ),
]

print("Loading embeddings + building FAQ index (first run may take a moment)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
retriever = Chroma.from_documents(
    FAQ_DOCS, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})

llm = init_chat_model("ollama:llama3.2", temperature=0)

faq_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Cosmic Learning FAQ bot. Use ONLY the context. "
            "If unknown, say you don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
faq_chain = faq_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class TicketCreate(BaseModel):
    message: str
    customer_email: str = "student@example.com"


class TicketLabel(BaseModel):
    category: Literal["billing", "technical", "sales", "general"] = Field(
        description="Support category"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent this is"
    )
    reason: str = Field(description="Short reason")


class TicketRecord(BaseModel):
    id: str
    customer_email: str
    message: str
    category: str
    urgency: str
    reason: str
    draft_reply: str
    status: str  # drafted | approved | rejected
    final_reply: str | None = None


class ReviewRequest(BaseModel):
    approved: bool
    edited_reply: str | None = None


# In-memory ticket store (real apps: Postgres)
TICKETS: dict[str, TicketRecord] = {}

classifier = llm.with_structured_output(TicketLabel)

draft_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Cosmic Learning support. Write a short professional reply. "
            "Do not invent policies. If unsure, ask one clarifying question.",
        ),
        (
            "human",
            "Category: {category}\nUrgency: {urgency}\nCustomer email:\n{message}",
        ),
    ]
)
draft_chain = draft_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cosmic Learning Support Desk", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def home():
    """Simple real UI so you can click and test like a product."""
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cosmic Learning Support Desk</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 880px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .sub { color: #555; margin-bottom: 24px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
    textarea, input { width: 100%; box-sizing: border-box; padding: 10px; margin: 8px 0 12px; }
    button { padding: 10px 14px; cursor: pointer; margin-right: 8px; }
    pre { background: #f6f8fa; padding: 12px; border-radius: 8px; white-space: pre-wrap; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; }
  </style>
</head>
<body>
  <h1>Cosmic Learning Support Desk</h1>
  <p class="sub">demo7 — FAQ chat + ticket draft + human review</p>

  <div class="card">
    <h2>1) FAQ Chat (RAG)</h2>
    <textarea id="question" rows="3" placeholder="Ask: How do I cancel? / How much is Premium?"></textarea>
    <button onclick="askFaq()">Ask FAQ</button>
    <pre id="faqOut">Answer will appear here...</pre>
  </div>

  <div class="card">
    <h2>2) Create Support Ticket</h2>
    <input id="email" value="maya@example.com" />
    <textarea id="ticketMsg" rows="4" placeholder="I was charged twice for Premium. Please refund."></textarea>
    <button onclick="createTicket()">Create + Draft Reply</button>
    <pre id="ticketOut">Ticket result will appear here...</pre>
  </div>

  <div class="card">
    <h2>3) Human Review (for high urgency)</h2>
    <input id="ticketId" placeholder="Paste ticket id from step 2" />
    <textarea id="editedReply" rows="4" placeholder="Optional edited reply before approve"></textarea>
    <div class="row">
      <button onclick="reviewTicket(true)">Approve</button>
      <button onclick="reviewTicket(false)">Reject</button>
    </div>
    <pre id="reviewOut">Review result will appear here...</pre>
  </div>

<script>
async function askFaq() {
  const question = document.getElementById('question').value;
  const res = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question})
  });
  const data = await res.json();
  document.getElementById('faqOut').textContent = JSON.stringify(data, null, 2);
}

async function createTicket() {
  const payload = {
    customer_email: document.getElementById('email').value,
    message: document.getElementById('ticketMsg').value
  };
  const res = await fetch('/tickets', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  document.getElementById('ticketOut').textContent = JSON.stringify(data, null, 2);
  if (data.id) document.getElementById('ticketId').value = data.id;
  if (data.draft_reply) document.getElementById('editedReply').value = data.draft_reply;
}

async function reviewTicket(approved) {
  const id = document.getElementById('ticketId').value;
  const edited_reply = document.getElementById('editedReply').value;
  const res = await fetch('/tickets/' + id + '/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({approved, edited_reply})
  });
  const data = await res.json();
  document.getElementById('reviewOut').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Real-world: in-app help chat endpoint."""
    docs = retriever.invoke(req.question)
    context = "\n".join(d.page_content for d in docs)
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    answer = faq_chain.invoke({"context": context, "question": req.question})
    return AskResponse(answer=answer, sources=sources)


@app.post("/tickets", response_model=TicketRecord)
def create_ticket(req: TicketCreate):
    """Real-world: customer submits a ticket → classify + draft."""
    label = classifier.invoke(
        "Classify this Cosmic Learning support message:\n" + req.message
    )
    draft = draft_chain.invoke(
        {
            "category": label.category,
            "urgency": label.urgency,
            "message": req.message,
        }
    )

    ticket = TicketRecord(
        id=str(uuid4())[:8],
        customer_email=req.customer_email,
        message=req.message,
        category=label.category,
        urgency=label.urgency,
        reason=label.reason,
        draft_reply=draft,
        status="drafted",
    )
    TICKETS[ticket.id] = ticket
    return ticket


@app.get("/tickets/{ticket_id}", response_model=TicketRecord)
def get_ticket(ticket_id: str):
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.post("/tickets/{ticket_id}/review", response_model=TicketRecord)
def review_ticket(ticket_id: str, req: ReviewRequest):
    """Real-world: support lead approves/edits before sending to customer."""
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if req.approved:
        ticket.final_reply = req.edited_reply or ticket.draft_reply
        ticket.status = "approved"
    else:
        ticket.final_reply = "Reply cancelled by reviewer."
        ticket.status = "rejected"

    TICKETS[ticket_id] = ticket
    return ticket


if __name__ == "__main__":
    print("\nOpen this in your browser: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""demo8 — Multi-turn Chat Widget API (real chatbot backend)

Real-world use:
  Embed this behind a website chat bubble, mobile app, or Intercom-like widget.
  Each visitor gets a session_id so the bot remembers earlier messages.

What you practice:
  - session memory (lesson35) stored per chat
  - FAQ RAG (lesson30/38) for grounded answers
  - FastAPI endpoints a frontend can call

Run:
  ollama serve
  python demo8.py

Open: http://127.0.0.1:8001
"""

from __future__ import annotations

from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge base
# Real world: sync from Notion / Zendesk Help Center / Markdown repo nightly.
# ---------------------------------------------------------------------------
DOCS = [
    Document(
        page_content="Free plan needs no credit card and includes 4 skills practice.",
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content="Premium is $12/month with advanced AI speaking feedback.",
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content="Cancel in Billing Settings any time before renewal.",
        metadata={"source": "billing.md"},
    ),
    Document(
        page_content="Refunds within 30 days; paid in 7 business days after approval.",
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content="Reset password via Account Settings → Security.",
        metadata={"source": "account.md"},
    ),
]

print("demo8: building chat knowledge index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
retriever = Chroma.from_documents(
    DOCS, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})

llm = init_chat_model("ollama:llama3.2", temperature=0)

# session_id -> list of LangChain messages
# Real world: Redis / Postgres chat_messages table keyed by session_id
SESSIONS: dict[str, list] = {}


class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: list[str]


app = FastAPI(title="Cosmic Chat Widget API", version="1.0.0")


def get_context(question: str) -> tuple[str, list[str]]:
    """Retrieve FAQ snippets for the current user question."""
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    # Real world: return stable doc ids/URLs for UI citation chips
    sources = [d.metadata.get("source", "faq") for d in docs]
    return context, sources


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>demo8 — Chat Widget</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; }
    #log { border: 1px solid #ddd; border-radius: 10px; min-height: 280px; padding: 12px; background: #fafafa; }
    .msg { margin: 8px 0; }
    .user { color: #0b57d0; }
    .bot { color: #0f7b3a; }
    .row { display: flex; gap: 8px; margin-top: 12px; }
    input { flex: 1; padding: 10px; }
    button { padding: 10px 14px; }
    code { background: #eee; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>demo8 — Multi-turn Chat Widget</h1>
  <p>Session: <code id="sid">starting...</code></p>
  <div id="log"></div>
  <div class="row">
    <input id="input" placeholder="Ask about Premium, refunds, password..." />
    <button onclick="send()">Send</button>
  </div>
<script>
let sessionId = null;
const log = document.getElementById('log');

function add(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = (role === 'user' ? 'You: ' : 'Bot: ') + text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function start() {
  const res = await fetch('/sessions', { method: 'POST' });
  const data = await res.json();
  sessionId = data.session_id;
  document.getElementById('sid').textContent = sessionId;
  add('bot', data.message);
}
async function send() {
  const input = document.getElementById('input');
  const message = input.value.trim();
  if (!message || !sessionId) return;
  add('user', message);
  input.value = '';
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session_id: sessionId, message })
  });
  const data = await res.json();
  add('bot', data.reply + (data.sources?.length ? '  [' + data.sources.join(', ') + ']' : ''));
}
start();
document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});
</script>
</body>
</html>
"""


@app.post("/sessions", response_model=StartSessionResponse)
def start_session():
    """Create a new chat session (like opening the chat bubble)."""
    session_id = str(uuid4())[:8]

    # System prompt is the "personality + rules" for this chat
    SESSIONS[session_id] = [
        SystemMessage(
            "You are Cosmic Learning's website chat assistant. "
            "Use the provided CONTEXT for product facts. "
            "Remember details the user shared earlier in this chat. "
            "If context is insufficient, say you don't know. Keep replies short."
        )
    ]
    return StartSessionResponse(
        session_id=session_id,
        message="Hi! Ask me about plans, refunds, or account help.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send one user message inside an existing session."""
    history = SESSIONS.get(req.session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call POST /sessions first.")

    # 1) Retrieve FAQ context for THIS question
    context, sources = get_context(req.message)

    # 2) Append user turn (include context so the model stays grounded)
    history.append(
        HumanMessage(
            f"CONTEXT:\n{context}\n\nUSER MESSAGE:\n{req.message}"
        )
    )

    # 3) Model reply using full history (memory)
    ai = llm.invoke(history)
    history.append(AIMessage(ai.content))

    # Real world: also persist history to DB here
    SESSIONS[req.session_id] = history

    return ChatResponse(
        session_id=req.session_id,
        reply=ai.content,
        sources=sorted(set(sources)),
    )


@app.get("/sessions/{session_id}/history")
def history(session_id: str):
    """Debug endpoint: inspect remembered messages."""
    msgs = SESSIONS.get(session_id)
    if msgs is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {
        "session_id": session_id,
        "turns": [
            {"type": m.__class__.__name__, "content": getattr(m, "content", "")}
            for m in msgs
        ],
    }


if __name__ == "__main__":
    print("\nOpen: http://127.0.0.1:8001\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)

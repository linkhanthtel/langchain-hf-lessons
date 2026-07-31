"""lesson18 — End-to-end RAG pipeline (Cosmic Learning knowledge base).

Pieces you already learned, now connected:

  knowledge docs
       ↓
  chunk + metadata
       ↓
  embed + store in Chroma     ← lesson5 / lesson9
       ↓
  retrieve top-k chunks       ← lesson3 / lesson10
       ↓
  generate answer with LLM    ← grounded on retrieved context only

Run:
  ollama serve
  python lesson18.py
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = init_chat_model("ollama:llama3.2", temperature=0)


# --- 1) Knowledge base (in real apps: PDFs, Notion, DB rows) ---
RAW_DOCS = [
    Document(
        page_content=(
            "Cosmic Learning free plan: all learners can practice reading, "
            "writing, listening, and speaking at no cost. No credit card required."
        ),
        metadata={"source": "pricing.md", "department": "billing", "doc_type": "policy"},
    ),
    Document(
        page_content=(
            "Cosmic Learning Premium unlocks advanced AI feedback, unlimited "
            "speaking practice, and personalized study plans for $12/month."
        ),
        metadata={"source": "pricing.md", "department": "billing", "doc_type": "policy"},
    ),
    Document(
        page_content=(
            "Refunds: customers can request a refund within 30 days of purchase. "
            "Approved refunds are processed within 7 business days."
        ),
        metadata={"source": "refunds.md", "department": "billing", "doc_type": "policy"},
    ),
    Document(
        page_content=(
            "Password reset: open Account Settings → Security → Reset Password. "
            "A reset link is emailed within a few minutes."
        ),
        metadata={"source": "account.md", "department": "support", "doc_type": "howto"},
    ),
    Document(
        page_content=(
            "Speaking practice: choose a prompt, record your answer, then review "
            "AI pronunciation and fluency feedback on the Results screen."
        ),
        metadata={"source": "features.md", "department": "product", "doc_type": "howto"},
    ),
]


def build_vectorstore(docs: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


vectorstore = build_vectorstore(RAW_DOCS)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


class RAGState(TypedDict):
    question: str
    context: str
    sources: list[str]
    answer: str


def retrieve_node(state: RAGState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    return {"context": context, "sources": sources}


def generate_node(state: RAGState):
    result = llm.invoke(
        [
            SystemMessage(
                "You are Cosmic Learning's support assistant. "
                "Answer ONLY using the provided context. "
                "If the context is insufficient, say you don't know. "
                "Be concise."
            ),
            HumanMessage(
                f"Context:\n{state['context']}\n\n"
                f"Question: {state['question']}\n\n"
                "Answer:"
            ),
        ]
    )
    return {"answer": result.content}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()


if __name__ == "__main__":
    questions = [
        "How do I reset my password?",
        "How long do refunds take?",
        "What is included in the free plan?",
        "Do you offer airport pickup?",  # should say don't know
    ]

    for question in questions:
        result = app.invoke(
            {
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
            }
        )
        print("=" * 60)
        print("Q:", question)
        print("Sources:", result["sources"])
        print("A:", result["answer"])
        print()

"""lesson21 — Agentic RAG with create_agent (LangChain docs pattern).

Docs reference:
  https://docs.langchain.com/oss/python/langgraph/agentic-rag
  https://docs.langchain.com/oss/python/langchain/agents

Difference from lesson18 (fixed RAG pipeline):
  lesson18: ALWAYS retrieve → then generate
  lesson21: the AGENT decides whether retrieval is needed

  User question
       │
       V
  create_agent loop
       |- answer directly (small talk / known facts)
       |- call retrieve_docs tool → then answer with context

Real-world scenario:
  A SaaS in-app help chatbot (Cosmic Learning).
  - "hi" -> no retrieval needed
  - "how do refunds work?" → must search the knowledge base
  - Saves cost/latency by not hitting the vector DB on every message

Run:
  ollama serve
  python lesson21.py
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Real-world: these docs usually come from Notion / Zendesk / uploaded PDFs
# and are re-indexed on a schedule (nightly job or webhook on publish).
RAW_DOCS = [
    Document(
        page_content=(
            "Cosmic Learning free plan includes reading, writing, listening, "
            "and speaking practice. No credit card required."
        ),
        metadata={"source": "pricing.md", "product": "cosmic-learning"},
    ),
    Document(
        page_content=(
            "Premium is $12/month and adds advanced AI feedback, unlimited "
            "speaking practice, and personalized study plans."
        ),
        metadata={"source": "pricing.md", "product": "cosmic-learning"},
    ),
    Document(
        page_content=(
            "Refunds can be requested within 30 days of purchase. "
            "Approved refunds are processed within 7 business days."
        ),
        metadata={"source": "refunds.md", "product": "cosmic-learning"},
    ),
    Document(
        page_content=(
            "Password reset: Account Settings → Security → Reset Password. "
            "The reset email usually arrives within a few minutes."
        ),
        metadata={"source": "account.md", "product": "cosmic-learning"},
    ),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_retriever():
    # Real-world: use a persistent Chroma/pgvector collection, not in-memory.
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=280, chunk_overlap=40
    ).split_documents(RAW_DOCS)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


@tool
def retrieve_docs(query: str) -> str:
    """Search Cosmic Learning help docs for pricing, refunds, and account help.

    Use this when the user asks about product policies or how-to steps.
    Do not use this for greetings or unrelated chit-chat.
    """
    # Real-world: log query + retrieved doc IDs for analytics / eval datasets.
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in docs
    )


agent = create_agent(
    # Docs: model can be a provider string like "ollama:llama3.2"
    model="ollama:llama3.2",
    tools=[retrieve_docs],
    system_prompt=(
        "You are Cosmic Learning's in-app help assistant.\n"
        "Rules:\n"
        "1. Greetings/thanks → reply in one short friendly sentence. No tools.\n"
        "2. Product/policy/how-to questions about Cosmic Learning → "
        "call retrieve_docs, then answer from that context only.\n"
        "3. Unrelated topics (sports, news, etc.) → politely say you only "
        "help with Cosmic Learning. Do not call tools.\n"
        "4. If docs are insufficient, say you don't know."
    ),
)


if __name__ == "__main__":
    # Real-world: each chat session would pass conversation history in messages.
    test_turns = [
        "Hi there!",
        "How do refunds work?",
        "What is included in the free plan?",
        "Who won the World Cup?",  # should not invent Cosmic Learning facts
    ]

    for text in test_turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": text}]}
        )
        print("=" * 60)
        print("USER:", text)
        print("ASSISTANT:", result["messages"][-1].content)
        # Useful while learning: see whether the agent called tools
        tool_calls = [
            m for m in result["messages"] if getattr(m, "tool_calls", None)
        ]
        print("TOOL CALLS:", bool(tool_calls))
        print()

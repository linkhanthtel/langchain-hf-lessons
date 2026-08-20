"""lesson34 - RAG + agent (search docs only when needed)

GOAL (one idea only):
  Combine lesson30 (RAG) with lesson33 (agent).

The agent has ONE retrieval tool.
It decides when to search the help docs.

Useful:
  - in-app help chatbot
  - "ask my FAQ" assistant

Builds on: lesson30 + lesson33
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

docs = [
    Document(page_content="Free plan: no credit card required."),
    Document(page_content="Premium is $12/month with advanced AI feedback."),
    Document(page_content="Refunds are processed within 7 business days."),
    Document(page_content="Reset password in Account Settings → Security."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
retriever = Chroma.from_documents(
    docs, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})


@tool
def search_help(query: str) -> str:
    """Search Cosmic Learning help docs for pricing, refunds, and account help."""
    found = retriever.invoke(query)
    if not found:
        return "No docs found."
    return "\n".join(doc.page_content for doc in found)


agent = create_agent(
    model="ollama:llama3.2",
    tools=[search_help],
    system_prompt=(
        "You help with Cosmic Learning. "
        "For product questions, call search_help first. "
        "For greetings, reply briefly without tools."
    ),
)

questions = [
    "Hi!",
    "How do I reset my password?",
    "How long do refunds take?",
]

for q in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print("=" * 50)
    print("Q:", q)
    print("A:", result["messages"][-1].content)
    print()

# Remember:
# lesson30 RAG = YOU always retrieve, then ask LLM
# lesson34 agentic RAG = AGENT chooses when to retrieve

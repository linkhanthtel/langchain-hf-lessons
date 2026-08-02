"""lesson20 — Agentic / corrective RAG.

Naïve RAG always retrieves once, then answers.
Corrective RAG adds a quality loop:

  START
    -> retrieve
    -> grade_documents (are chunks actually relevant?)
    -> route 
         |- generate (good enough)
         |- rewrite_question (bad retrieval) -> retrieve again
    -> generate
    -> END

This is closer to production support bots that refuse to hallucinate
when retrieval fails.
"""

from typing import Literal, TypedDict

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

RAW_DOCS = [
    Document(
        page_content=(
            "Cosmic Learning free plan includes reading, writing, listening, "
            "and speaking practice with no credit card required."
        ),
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content=(
            "Premium is $12/month and adds advanced AI feedback plus study plans."
        ),
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content=(
            "Refunds can be requested within 30 days and are paid in 7 business days."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content=(
            "Password reset is available under Account Settings → Security."
        ),
        metadata={"source": "account.md"},
    ),
]


def build_vectorstore() -> Chroma:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=260, chunk_overlap=40
    ).split_documents(RAW_DOCS)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


vectorstore = build_vectorstore()


class CorrectiveRAGState(TypedDict):
    question: str
    documents: list[str]
    sources: list[str]
    grades: list[str]
    retrieval_ok: bool
    rewrite_count: int
    answer: str


def retrieve_node(state: CorrectiveRAGState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {
        "documents": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", "unknown") for doc in docs],
    }


def grade_documents_node(state: CorrectiveRAGState):
    """Keep only chunks that look relevant enough.

    Production systems often use an LLM grader here.
    This lesson uses distance scores so results stay stable while learning.
    Lower distance = more similar.
    """
    scored = vectorstore.similarity_search_with_score(state["question"], k=4)
    # Chroma L2 distance: smaller is better. Keep reasonably close matches.
    max_distance = 1.35

    grades: list[str] = []
    relevant_docs: list[str] = []
    sources: list[str] = []

    for doc, distance in scored:
        grade = "yes" if distance <= max_distance else "no"
        grades.append(f"{grade}(dist={distance:.2f})")
        if grade == "yes":
            relevant_docs.append(doc.page_content)
            sources.append(doc.metadata.get("source", "unknown"))

    return {
        "grades": grades,
        "documents": relevant_docs,
        "sources": sources,
        "retrieval_ok": len(relevant_docs) > 0,
    }


def route_after_grade(
    state: CorrectiveRAGState,
) -> Literal["generate", "rewrite_question"]:
    if state["retrieval_ok"]:
        return "generate"
    # Avoid infinite loops
    if state["rewrite_count"] >= 1:
        return "generate"
    return "rewrite_question"


def rewrite_question_node(state: CorrectiveRAGState):
    result = llm.invoke(
        [
            SystemMessage(
                "Rewrite the question to be clearer for searching a product FAQ. "
                "Return only the rewritten question."
            ),
            HumanMessage(state["question"]),
        ]
    )
    return {
        "question": result.content.strip(),
        "rewrite_count": state["rewrite_count"] + 1,
        "documents": [],
        "sources": [],
        "grades": [],
        "retrieval_ok": False,
    }


def generate_node(state: CorrectiveRAGState):
    if not state["documents"]:
        return {
            "answer": (
                "I don't have enough information in the knowledge base "
                "to answer that."
            )
        }

    context = "\n\n".join(state["documents"])
    result = llm.invoke(
        [
            SystemMessage(
                "You are Cosmic Learning support. "
                "Use the context below as ground truth and answer directly. "
                "Do not say you are unsure if the context contains the answer. "
                "Keep the answer to 1-2 sentences."
            ),
            HumanMessage(
                f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
            ),
        ]
    )
    return {"answer": result.content}


graph = StateGraph(CorrectiveRAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("rewrite_question", rewrite_question_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grade)
graph.add_edge("rewrite_question", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()


if __name__ == "__main__":
    cases = [
        "How do refunds work on Cosmic Learning?",
        "How can I reset my Cosmic Learning password?",
        "What is the weather in Singapore?",  # off-topic → should refuse
    ]

    for question in cases:
        result = app.invoke(
            {
                "question": question,
                "documents": [],
                "sources": [],
                "grades": [],
                "retrieval_ok": False,
                "rewrite_count": 0,
                "answer": "",
            }
        )
        print("=" * 60)
        print("Q:", question)
        print("Final question used:", result["question"])
        print("Grades:", result["grades"])
        print("Rewrite count:", result["rewrite_count"])
        print("A:", result["answer"])
        print()

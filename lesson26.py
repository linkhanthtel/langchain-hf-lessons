"""lesson26 — Groundedness check (anti-hallucination gate).

Docs reference:
  RAG generate → verify pattern (related to corrective RAG / lesson20)

LESSON vs REAL WORLD
-----------------------------------------------------------------------------
This lesson                          Real products
-----------------------------------  -----------------------------------------
retrieve → draft → grade             Support / healthcare / finance copilots
YES/NO groundedness score            Separate judge model or NLI classifier
block ungrounded answers             Show "I don't know" + escalate to human
Cosmic Learning FAQ corpus           Approved policy corpus only
local Ollama judge                   Stronger judge model than the drafter

Useful cases:
  - Banks / insurance: cannot invent policy numbers
  - Healthcare admin bots: cannot invent clinical guidance
  - HR policy assistants: cite handbook sections only
  - Student-facing edtech tutors: don't invent syllabus deadlines

Compare with earlier lessons:
  lesson18: retrieve → answer (trusts the LLM)
  lesson20: rewrite query if retrieval is weak
  lesson23: force citations in structured output
  lesson26: draft an answer, then VERIFY it is supported by context
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
# Real-world: often use a cheaper model to draft and a stronger model to judge.
drafter = init_chat_model("ollama:llama3.2", temperature=0)
judge = init_chat_model("ollama:llama3.2", temperature=0)

RAW_DOCS = [
    Document(
        page_content=(
            "Refunds can be requested within 30 days of purchase and are "
            "processed within 7 business days after approval."
        ),
        metadata={"source": "refunds.md"},
    ),
    Document(
        page_content=(
            "Premium is $12 per month and includes advanced AI feedback."
        ),
        metadata={"source": "pricing.md"},
    ),
    Document(
        page_content=(
            "Password resets are available in Account Settings → Security."
        ),
        metadata={"source": "account.md"},
    ),
]

retriever = Chroma.from_documents(
    documents=RecursiveCharacterTextSplitter(
        chunk_size=240, chunk_overlap=40
    ).split_documents(RAW_DOCS),
    embedding=embeddings,
).as_retriever(search_kwargs={"k": 3})


class GroundedState(TypedDict):
    question: str
    context: str
    draft_answer: str
    grounded: bool
    final_answer: str
    judge_reason: str


def retrieve_node(state: GroundedState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}" for d in docs
    )
    return {"context": context}


def draft_node(state: GroundedState):
    # Real-world: this is what users would see if you skipped verification.
    result = drafter.invoke(
        [
            SystemMessage(
                "Answer using the context. If unsure, still attempt a helpful reply."
            ),
            HumanMessage(
                f"Context:\n{state['context']}\n\nQuestion: {state['question']}"
            ),
        ]
    )
    return {"draft_answer": result.content}


def judge_node(state: GroundedState):
    """Check whether every claim in the draft is supported by context."""
    # Real-world: many teams use a dedicated evaluation prompt or NLI model.
    result = judge.invoke(
        [
            SystemMessage(
                "You are a strict groundedness judge.\n"
                "Reply in this exact format:\n"
                "GROUNDED: YES|NO\n"
                "REASON: <short reason>\n"
                "Mark NO if the draft adds facts not present in the context."
            ),
            HumanMessage(
                f"Context:\n{state['context']}\n\n"
                f"Question: {state['question']}\n\n"
                f"Draft answer:\n{state['draft_answer']}"
            ),
        ]
    )
    text = result.content
    grounded = "GROUNDED: YES" in text.upper() or (
        "YES" in text.upper().splitlines()[0]
    )
    return {
        "grounded": grounded,
        "judge_reason": text.strip(),
    }


def route_after_judge(
    state: GroundedState,
) -> Literal["accept_answer", "reject_answer"]:
    return "accept_answer" if state["grounded"] else "reject_answer"


def accept_answer(state: GroundedState):
    # Real-world: send to chat UI + store citations in analytics.
    return {"final_answer": state["draft_answer"]}


def reject_answer(state: GroundedState):
    # Real-world: escalate to human or ask a clarifying question.
    return {
        "final_answer": (
            "I don't have enough verified information in our help docs "
            "to answer that safely. Please contact support@cosmiclearning.example."
        )
    }


graph = StateGraph(GroundedState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("draft", draft_node)
graph.add_node("judge", judge_node)
graph.add_node("accept_answer", accept_answer)
graph.add_node("reject_answer", reject_answer)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "draft")
graph.add_edge("draft", "judge")
graph.add_conditional_edges("judge", route_after_judge)
graph.add_edge("accept_answer", END)
graph.add_edge("reject_answer", END)

app = graph.compile()


if __name__ == "__main__":
    cases = [
        # Should be grounded in refunds.md
        "How long do refunds take after approval?",
        # Should be grounded in pricing.md
        "How much is Premium?",
        # Likely ungrounded / rejected — not in corpus
        "Do you offer free airport pickup for Premium members?",
    ]

    print("Pipeline: retrieve → draft → judge → accept/reject\n")

    for question in cases:
        result = app.invoke(
            {
                "question": question,
                "context": "",
                "draft_answer": "",
                "grounded": False,
                "final_answer": "",
                "judge_reason": "",
            }
        )
        print("=" * 60)
        print("Q:", question)
        print("DRAFT:", result["draft_answer"])
        print("JUDGE:", result["judge_reason"].replace("\n", " | "))
        print("GROUNDED:", result["grounded"])
        print("FINAL:", result["final_answer"])
        print()

"""demo6 — Human-in-the-loop customer support draft.

Flow:
1. LLM drafts a reply from the customer email
2. Graph pauses for human review (approve / edit / reject)
3. Resume and print the final reply
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

llm = init_chat_model("ollama:llama3.2", temperature=0)


class SupportState(TypedDict):
    email_content: str
    draft_response: str
    final_response: str
    status: str

def draft_node(state: SupportState):
    """LLM writes a first draft from the customer email."""
    result = llm.invoke(
        [
            SystemMessage(
                "You are a polite customer support agent. "
                "Write a short, professional reply. Do not invent policies."
            ),
            HumanMessage(f"Customer email:\n{state['email_content']}"),
        ]
    )
    return {
        "draft_response": result.content,
        "status": "drafted",
    }


def human_review_node(state: SupportState):
    """Pause here until a human approves, edits, or rejects the draft."""
    decision = interrupt(
        {
            "question": "Approve this draft, edit it, or reject it?",
            "draft_response": state["draft_response"],
        }
    )

    if decision.get("approved"):
        edited = decision.get("edited_response") or state["draft_response"]
        return {
            "final_response": edited,
            "status": "approved",
        }

    return {
        "final_response": "Reply cancelled by reviewer.",
        "status": "rejected",
    }


app = (
    StateGraph(SupportState)
    .add_node("draft", draft_node)
    .add_node("human_review", human_review_node)
    .add_edge(START, "draft")
    .add_edge("draft", "human_review")
    .add_edge("human_review", END)
    .compile(checkpointer=InMemorySaver())
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "support-001"}}

    # Step 1: run until the human-review interrupt
    paused = app.invoke(
        {
            "email_content": (
                "I was charged twice for my Cosmic Learning subscription! "
                "Please refund the extra charge."
            ),
            "draft_response": "",
            "final_response": "",
            "status": "new",
        },
        config,
    )

    print("=== Interrupted for human review ===")
    print("Draft:\n", paused["draft_response"])
    print("Interrupt payload:", paused.get("__interrupt__"))

    # Step 2: human approves (optionally with edits), then resume
    resumed = app.invoke(
        Command(
            resume={
                "approved": True,
                "edited_response": (
                    "We're sorry about the double charge. "
                    "We've started a full refund — it should arrive in 5–7 business days."
                ),
            }
        ),
        config,
    )

    print("\n=== Final result ===")
    print("Status:", resumed["status"])
    print("Final reply:\n", resumed["final_response"])

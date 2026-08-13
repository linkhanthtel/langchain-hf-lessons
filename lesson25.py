"""lesson25 — Intent router → specialized handlers (contact-center style).

Docs reference:
  https://docs.langchain.com/oss/python/langgraph/overview
  Conditional edges + multi-node workflows

LESSON vs REAL WORLD
-----------------------------------------------------------------------------
This lesson                          Real products
-----------------------------------  -----------------------------------------
rule/LLM intent labels               NLU classifier / LLM router service
handle_billing node                  Billing microservice + Stripe tools
handle_tech node                     Statuspage + log search tools
handle_sales node                    CRM + pricing catalog RAG
escalate node                        Zendesk ticket create + human queue
one Python graph                     API gateway -> worker queues

Useful cases:
  - Phone IVR / chat "Press 1 for billing..."
  - Marketplace support (buyer vs seller vs shipping)
  - Internal IT helpdesk (access, hardware, software)
  - Bank assistants (cards, transfers, fraud — different compliance paths)

Why not one giant agent with every tool?
  - Safer permissions (sales tools ≠ refund tools)
  - Clearer evals per department
  - Cheaper prompts (smaller tool lists)
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RouterState(TypedDict):
    message: str
    intent: str
    department: str
    reply: str
    escalate: bool


def classify_intent(state: RouterState):
    """Lightweight router.

    Real-world options:
      1) fine-tuned classifier (fast/cheap)
      2) LLM JSON classify (flexible)
      3) embeddings nearest-intent labels
    """
    text = state["message"].lower()

    if any(w in text for w in ["refund", "invoice", "charge", "payment", "billing"]):
        intent, department = "billing", "billing"
    elif any(w in text for w in ["login", "password", "crash", "bug", "error", "down"]):
        intent, department = "technical", "engineering"
    elif any(w in text for w in ["price", "premium", "upgrade", "discount", "plan"]):
        intent, department = "sales", "sales"
    elif any(w in text for w in ["lawyer", "lawsuit", "police", "threat"]):
        # Real-world: sensitive intents skip bots and go straight to humans.
        intent, department = "sensitive", "trust_and_safety"
    else:
        intent, department = "general", "support"

    return {"intent": intent, "department": department}


def route_intent(
    state: RouterState,
) -> Literal["handle_billing", "handle_technical", "handle_sales", "escalate", "handle_general"]:
    mapping = {
        "billing": "handle_billing",
        "technical": "handle_technical",
        "sales": "handle_sales",
        "sensitive": "escalate",
    }
    return mapping.get(state["intent"], "handle_general")


def handle_billing(state: RouterState):
    # Real-world: call Stripe + refund policy RAG tools here.
    return {
        "escalate": False,
        "reply": (
            "Billing desk: I can help with charges, invoices, and refunds. "
            "Please share your order ID (ORD-xxxx) if you have one."
        ),
    }


def handle_technical(state: RouterState):
    # Real-world: query Statuspage + error logs + known-issue KB.
    return {
        "escalate": False,
        "reply": (
            "Tech desk: sorry you're hitting an issue. "
            "Tell me your device/OS and whether login or the lesson player fails."
        ),
    }


def handle_sales(state: RouterState):
    # Real-world: pricing RAG + CRM lead capture.
    return {
        "escalate": False,
        "reply": (
            "Sales desk: Premium is $12/month with advanced AI feedback. "
            "I can compare Free vs Premium for your goal."
        ),
    }


def handle_general(state: RouterState):
    return {
        "escalate": False,
        "reply": (
            "General support: I can help with Cosmic Learning account, "
            "billing, technical issues, or plans. What do you need?"
        ),
    }


def escalate(state: RouterState):
    # Real-world: create high-priority ticket, page on-call, freeze auto-refunds.
    return {
        "escalate": True,
        "reply": (
            "This looks sensitive. I'm connecting you to a human specialist "
            "and pausing automated actions on your account."
        ),
    }


graph = StateGraph(RouterState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("handle_billing", handle_billing)
graph.add_node("handle_technical", handle_technical)
graph.add_node("handle_sales", handle_sales)
graph.add_node("handle_general", handle_general)
graph.add_node("escalate", escalate)

graph.add_edge(START, "classify_intent")
graph.add_conditional_edges("classify_intent", route_intent)
graph.add_edge("handle_billing", END)
graph.add_edge("handle_technical", END)
graph.add_edge("handle_sales", END)
graph.add_edge("handle_general", END)
graph.add_edge("escalate", END)

app = graph.compile()


if __name__ == "__main__":
    cases = [
        "I was charged twice, need a refund",
        "The app crashes when I open speaking practice",
        "Is Premium worth upgrading for IELTS?",
        "I will sue you and call my lawyer",
        "Hello, what can you help with?",
    ]

    print("Compare to real contact centers:")
    print("  Router = skills-based routing")
    print("  escalate = priority human queue\n")

    for message in cases:
        result = app.invoke(
            {
                "message": message,
                "intent": "",
                "department": "",
                "reply": "",
                "escalate": False,
            }
        )
        print("=" * 60)
        print("MSG:", message)
        print(
            f"ROUTE: intent={result['intent']} "
            f"dept={result['department']} escalate={result['escalate']}"
        )
        print("REPLY:", result["reply"])
        print()

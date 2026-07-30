"""lesson16 — Real-world ticket triage with complex multi-node routing.

Cosmic Learning support desk pipeline:

  START
    → parse_ticket        (normalize customer message)
    → classify_intent     (refund / technical / billing / general)
    → extract_entities    (order_id, amount, product)
    → route by intent
         ├─ handle_refund
         ├─ handle_technical
         ├─ handle_billing
         └─ handle_general
    → compose_reply
    → END

This lesson focuses on conditional edges + shared state across many nodes.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    raw_message: str
    cleaned_message: str
    intent: str
    order_id: str | None
    amount: float | None
    product: str | None
    handler_notes: str
    priority: str
    reply: str


def parse_ticket(state: TicketState):
    """Clean and normalize the incoming support message."""
    cleaned = " ".join(state["raw_message"].split())
    return {
        "cleaned_message": cleaned,
        "priority": "normal",
    }


def classify_intent(state: TicketState):
    """Classify what the customer wants.

    In production, replace this with an LLM classifier.
    """
    text = state["cleaned_message"].lower()

    if any(word in text for word in ["refund", "charged twice", "money back"]):
        intent = "refund"
    elif any(word in text for word in ["login", "password", "crash", "bug", "error"]):
        intent = "technical"
    elif any(word in text for word in ["invoice", "receipt", "billing", "subscription"]):
        intent = "billing"
    else:
        intent = "general"

    priority = "high" if "urgent" in text or "twice" in text else "normal"
    return {"intent": intent, "priority": priority}


def extract_entities(state: TicketState):
    """Pull structured fields out of the free-text ticket."""
    text = state["cleaned_message"]
    order_id = None
    amount = None
    product = None

    for token in text.replace(",", " ").split():
        if token.upper().startswith("ORD-"):
            order_id = token.upper().rstrip(".,!?")
        if token.startswith("$"):
            try:
                amount = float(token[1:])
            except ValueError:
                pass

    lowered = text.lower()
    if "premium" in lowered:
        product = "premium"
    elif "cosmic learning" in lowered or "subscription" in lowered:
        product = "subscription"

    return {
        "order_id": order_id,
        "amount": amount,
        "product": product,
    }


def route_intent(
    state: TicketState,
) -> Literal["handle_refund", "handle_technical", "handle_billing", "handle_general"]:
    """Conditional edge: send the ticket to the right specialist node."""
    mapping = {
        "refund": "handle_refund",
        "technical": "handle_technical",
        "billing": "handle_billing",
    }
    return mapping.get(state["intent"], "handle_general")


def handle_refund(state: TicketState):
    order = state["order_id"] or "unknown order"
    amount = state["amount"]
    notes = (
        f"Refund desk reviewing {order}. "
        f"Claimed amount: {amount if amount is not None else 'not provided'}."
    )
    return {"handler_notes": notes}


def handle_technical(state: TicketState):
    return {
        "handler_notes": (
            "Technical desk: ask for device/OS, reproduce steps, "
            "and check known outages for Cosmic Learning."
        )
    }


def handle_billing(state: TicketState):
    return {
        "handler_notes": (
            "Billing desk: pull invoice history and confirm active subscription plan."
        )
    }


def handle_general(state: TicketState):
    return {
        "handler_notes": "General support: gather more details before escalating."
    }


def compose_reply(state: TicketState):
    """Final node: turn handler notes into a customer-facing reply."""
    reply = (
        f"Thanks for contacting Cosmic Learning Support.\n"
        f"We classified your request as: {state['intent']} "
        f"(priority={state['priority']}).\n"
        f"Internal notes: {state['handler_notes']}\n"
        f"Order: {state['order_id'] or 'n/a'} | "
        f"Product: {state['product'] or 'n/a'}"
    )
    return {"reply": reply}


graph = StateGraph(TicketState)

graph.add_node("parse_ticket", parse_ticket)
graph.add_node("classify_intent", classify_intent)
graph.add_node("extract_entities", extract_entities)
graph.add_node("handle_refund", handle_refund)
graph.add_node("handle_technical", handle_technical)
graph.add_node("handle_billing", handle_billing)
graph.add_node("handle_general", handle_general)
graph.add_node("compose_reply", compose_reply)

graph.add_edge(START, "parse_ticket")
graph.add_edge("parse_ticket", "classify_intent")
graph.add_edge("classify_intent", "extract_entities")
graph.add_conditional_edges("extract_entities", route_intent)
graph.add_edge("handle_refund", "compose_reply")
graph.add_edge("handle_technical", "compose_reply")
graph.add_edge("handle_billing", "compose_reply")
graph.add_edge("handle_general", "compose_reply")
graph.add_edge("compose_reply", END)

app = graph.compile()


if __name__ == "__main__":
    tickets = [
        "URGENT!! I was charged twice for ORD-1001, $89.50 on Cosmic Learning Premium.",
        "I cannot login to Cosmic Learning, password reset email never arrives.",
        "Please send the invoice/receipt for my subscription this month.",
        "Hello, do you offer student discounts?",
    ]

    for ticket in tickets:
        result = app.invoke(
            {
                "raw_message": ticket,
                "cleaned_message": "",
                "intent": "",
                "order_id": None,
                "amount": None,
                "product": None,
                "handler_notes": "",
                "priority": "",
                "reply": "",
            }
        )
        print("=" * 60)
        print("INPUT:", ticket)
        print("INTENT:", result["intent"], "| PRIORITY:", result["priority"])
        print("ENTITIES:", result["order_id"], result["amount"], result["product"])
        print("REPLY:\n", result["reply"])
        print()

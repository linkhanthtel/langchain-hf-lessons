"""lesson17 — Real-world refund workflow with risk gates + human review.

Cosmic Learning refund pipeline:

  START
    → parse_request
    → lookup_order          (mock order DB)
    → check_policy          (mock policy rules)
    → score_risk            (amount / status / timing)
    → route by risk
         ├─ auto_approve    (low risk)
         ├─ human_review    (medium/high — interrupt)
         └─ auto_reject     (policy fail)
    → notify_customer
    → END

Builds on lesson15/demo6 (HITL) and lesson12 (order + policy tools),
but as an explicit multi-node graph instead of a free-form agent.
"""

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


ORDERS = {
    "ORD-1001": {
        "status": "refund_processing",
        "amount": 89.50,
        "days_since_purchase": 12,
        "customer": "lin@example.com",
    },
    "ORD-1002": {
        "status": "delivered",
        "amount": 42.00,
        "days_since_purchase": 45,
        "customer": "maya@example.com",
    },
    "ORD-1003": {
        "status": "delivered",
        "amount": 320.00,
        "days_since_purchase": 5,
        "customer": "sam@example.com",
    },
}


class RefundState(TypedDict):
    order_id: str
    reason: str
    order: dict | None
    policy_ok: bool
    policy_notes: str
    risk_level: str
    decision: str
    refund_amount: float
    customer_message: str


def parse_request(state: RefundState):
    return {
        "order_id": state["order_id"].strip().upper(),
        "reason": state["reason"].strip(),
        "decision": "pending",
        "customer_message": "",
    }


def lookup_order(state: RefundState):
    order = ORDERS.get(state["order_id"])
    if not order:
        return {
            "order": None,
            "policy_ok": False,
            "policy_notes": f"Order {state['order_id']} not found.",
            "risk_level": "reject",
        }
    return {"order": order}


def check_policy(state: RefundState):
    order = state["order"]
    if order is None:
        return {}

    notes = []
    ok = True

    if order["days_since_purchase"] > 30:
        ok = False
        notes.append("Outside 30-day refund window.")
    if order["status"] == "refund_processing":
        notes.append("Refund already in progress.")
    if "fraud" in state["reason"].lower():
        ok = False
        notes.append("Fraud claims require security team, not auto-refund.")

    if ok and not notes:
        notes.append("Within policy window.")

    return {
        "policy_ok": ok,
        "policy_notes": " ".join(notes),
    }


def score_risk(state: RefundState):
    """Decide auto-approve vs human review vs reject."""
    if state["order"] is None or not state["policy_ok"]:
        return {"risk_level": "reject", "refund_amount": 0.0}

    amount = float(state["order"]["amount"])

    if amount >= 200:
        risk = "high"
    elif amount >= 75 or state["order"]["status"] == "refund_processing":
        risk = "medium"
    else:
        risk = "low"

    return {
        "risk_level": risk,
        "refund_amount": amount,
    }


def route_risk(
    state: RefundState,
) -> Literal["auto_approve", "human_review", "auto_reject"]:
    if state["risk_level"] == "low":
        return "auto_approve"
    if state["risk_level"] in {"medium", "high"}:
        return "human_review"
    return "auto_reject"


def auto_approve(state: RefundState):
    return {
        "decision": "approved",
        "customer_message": (
            f"Your refund of ${state['refund_amount']:.2f} for "
            f"{state['order_id']} has been approved. "
            "Expect 5–7 business days."
        ),
    }


def auto_reject(state: RefundState):
    return {
        "decision": "rejected",
        "refund_amount": 0.0,
        "customer_message": (
            f"We could not approve a refund for {state['order_id']}. "
            f"Reason: {state['policy_notes']}"
        ),
    }


def human_review(state: RefundState):
    """Pause for a support lead when risk is medium/high."""
    decision = interrupt(
        {
            "question": "Approve, reduce, or reject this refund?",
            "order_id": state["order_id"],
            "amount": state["refund_amount"],
            "risk_level": state["risk_level"],
            "policy_notes": state["policy_notes"],
            "reason": state["reason"],
        }
    )

    action = decision.get("action", "reject")
    if action == "approve":
        amount = float(decision.get("amount", state["refund_amount"]))
        return {
            "decision": "approved_by_human",
            "refund_amount": amount,
            "customer_message": (
                f"A support lead approved your refund of ${amount:.2f} "
                f"for {state['order_id']}. Expect 5–7 business days."
            ),
        }

    if action == "reduce":
        amount = float(decision.get("amount", state["refund_amount"] / 2))
        return {
            "decision": "partial_refund",
            "refund_amount": amount,
            "customer_message": (
                f"We approved a partial refund of ${amount:.2f} "
                f"for {state['order_id']}."
            ),
        }

    return {
        "decision": "rejected_by_human",
        "refund_amount": 0.0,
        "customer_message": (
            f"After review, we cannot refund {state['order_id']}. "
            "Please reply if you have more details."
        ),
    }


def notify_customer(state: RefundState):
    """Final side-effect style node (mock email/push notification)."""
    print(
        f"[notify] decision={state['decision']} "
        f"order={state['order_id']} amount={state['refund_amount']:.2f}"
    )
    return {}


graph = StateGraph(RefundState)

graph.add_node("parse_request", parse_request)
graph.add_node("lookup_order", lookup_order)
graph.add_node("check_policy", check_policy)
graph.add_node("score_risk", score_risk)
graph.add_node("auto_approve", auto_approve)
graph.add_node("human_review", human_review)
graph.add_node("auto_reject", auto_reject)
graph.add_node("notify_customer", notify_customer)

graph.add_edge(START, "parse_request")
graph.add_edge("parse_request", "lookup_order")
graph.add_edge("lookup_order", "check_policy")
graph.add_edge("check_policy", "score_risk")
graph.add_conditional_edges("score_risk", route_risk)
graph.add_edge("auto_approve", "notify_customer")
graph.add_edge("human_review", "notify_customer")
graph.add_edge("auto_reject", "notify_customer")
graph.add_edge("notify_customer", END)

app = graph.compile(checkpointer=InMemorySaver())


def run_case(thread_id: str, order_id: str, reason: str, human_action: dict | None = None):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "order_id": order_id,
            "reason": reason,
            "order": None,
            "policy_ok": False,
            "policy_notes": "",
            "risk_level": "",
            "decision": "",
            "refund_amount": 0.0,
            "customer_message": "",
        },
        config,
    )

    if result.get("__interrupt__") and human_action is not None:
        print("Interrupted for human review:", result["__interrupt__"])
        result = app.invoke(Command(resume=human_action), config)

    print("=" * 60)
    print(f"ORDER: {order_id}")
    print(f"RISK: {result['risk_level']} | DECISION: {result['decision']}")
    print(f"AMOUNT: {result['refund_amount']}")
    print(f"MESSAGE: {result['customer_message']}")
    print()
    return result


if __name__ == "__main__":
    # Case 1: low risk → auto approve (ORD-1002 is outside window → reject)
    run_case("t-low", "ORD-1002", "Changed my mind about the course")

    # Case 2: already processing + medium amount → human review
    run_case(
        "t-medium",
        "ORD-1001",
        "I was charged twice, please refund",
        human_action={"action": "approve", "amount": 89.50},
    )

    # Case 3: high amount → human review with partial refund
    run_case(
        "t-high",
        "ORD-1003",
        "Premium plan not useful for me",
        human_action={"action": "reduce", "amount": 160.00},
    )

    # Case 4: missing order → auto reject
    run_case("t-missing", "ORD-9999", "Please refund")

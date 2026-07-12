from langchain.tools import tool
from langchain.agents import create_agent

@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of a customer order."""

    # Replace this mock data with a PostgreSQL query.
    orders = {
        "ORD-1001": {
            "status": "refund_processing",
            "refund_requested_at": "2026-07-8",
            "amount": 89.50,
        },
        "ORD-1002": {
            "status": "delivered",
            "refund_requested_at": None,
            "amount": 42.00,
        },
    }

    order = orders.get(order_id)

    if not order:
        return f"No order was found for {order_id}."

    return (
        f"Order {order_id}: status={order['status']}, "
        f"refund_requested_at={order['refund_requested_at']}, "
        f"amount={order['amount']}"
    )


@tool
def search_refund_policy(question: str) -> str:
    """Search the company's refund policy knowledge base."""

    # Replace this with pgvector or another RAG retriever.
    return (
        "Approved refunds are normally processed within "
        "seven business days after the refund request."
    )


agent = create_agent(
    model="ollama:llama3.2",
    tools=[
        get_order_status,
        search_refund_policy,
    ],
    system_prompt=(
        "You are a customer support agent. "
        "Use tools to verify order status and company policy. "
        "Do not invent order information. "
        "Clearly state when information cannot be verified."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "My order is ORD-1001. "
                    "Where is my refund, and how long should it take?"
                ),
            }
        ]
    }
)

print(result["messages"][-1].content)
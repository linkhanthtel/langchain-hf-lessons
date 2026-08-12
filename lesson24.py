"""lesson24 — Multi-turn memory agent (conversation continuity).

Docs reference:
  https://docs.langchain.com/oss/python/langchain/agents
  (create_agent + checkpointer / thread_id)

LESSON vs REAL WORLD
-----------------------------------------------------------------------------
This lesson                          Real products
-----------------------------------  -----------------------------------------
InMemorySaver                        Redis / Postgres checkpointer
thread_id="user-42"                  Intercom/Zendesk conversation ID
short chat history in state          30-day ticket history + CRM notes
one local Ollama model               GPT/Claude + fallback models

Useful cases:
  - In-app chatbot that remembers "my order is ORD-1001" from earlier turns
  - WhatsApp / Slack support bots
  - Onboarding wizards ("what is your goal?" -> later personalize advice)

Without memory, every message is a cold start — bad UX and repeated tool calls.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Real-world: replace with CRM / order-service API.
ORDER_DB = {
    "ORD-1001": {"status": "shipped", "eta_days": 2, "item": "Premium annual plan"},
    "ORD-1002": {"status": "refund_processing", "eta_days": 5, "item": "Premium monthly"},
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up order status by ID (example: ORD-1001)."""
    # Real-world: authenticated API call scoped to the logged-in customer.
    order = ORDER_DB.get(order_id.strip().upper())
    if not order:
        return f"No order found for {order_id}."
    return (
        f"{order_id.upper()}: status={order['status']}, "
        f"item={order['item']}, eta_days={order['eta_days']}"
    )


# Real-world: use a durable checkpointer so restarts don't wipe chats.
memory = InMemorySaver()

agent = create_agent(
    model="ollama:llama3.2",
    tools=[get_order_status],
    system_prompt=(
        "You are Cosmic Learning support. "
        "Remember details the user already shared in this thread. "
        "If they mention an order ID, call get_order_status. "
        "Keep replies short."
    ),
    checkpointer=memory,
)


if __name__ == "__main__":
    # Real-world: thread_id = browser session / ticket ID / user ID.
    config = {"configurable": {"thread_id": "customer-lin-001"}}

    turns = [
        "Hi, my order is ORD-1001.",
        "What's the status?",  # no order ID repeated — memory must help
        "When should it arrive?",
    ]

    for text in turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config,
        )
        print("=" * 60)
        print("USER:", text)
        print("BOT:", result["messages"][-1].content)
        print()

    print(
        "Compare: without thread_id/memory, turn 2 would ask again "
        "for the order ID. With memory, the bot should reuse ORD-1001."
    )

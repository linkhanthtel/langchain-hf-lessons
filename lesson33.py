"""lesson33 - Tiny agent (model + tool loop, done for you)

GOAL (one idea only):
  Let create_agent run the tool loop automatically.

Flow:
  user question
    -> model may call a tool
    -> tool runs
    -> model writes the final answer

Useful:
  - support bot that looks up orders
  - assistant that checks weather / docs / prices

Builds on: lesson32
Compare:
  lesson32 = you see the tool request
  lesson33 = LangChain runs the full loop
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool
def get_plan_price(plan_name: str) -> str:
    """Return the monthly price for a Cosmic Learning plan (free or premium)."""
    prices = {"free": "$0 per month", "premium": "$9 per month"}
    return prices.get(plan_name.lower().strip(), "Unknown plan")


@tool
def get_refund_days() -> str:
    """Return how many business days an approved refund takes."""
    return "Approved refunds take 7 business days."


agent = create_agent(
    model="ollama:llama3.2",
    tools=[get_plan_price, get_refund_days],
    system_prompt=(
        "You are Cosmic Learning support. "
        "Use tools for prices and refund timing. Keep answers short."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How much is Premium, and how long do refunds take?",
            }
        ]
    }
)

print(result["messages"][-1].content)

# Remember:
# create_agent = model + tools + loop until done
# You pass messages in, you read the last message out

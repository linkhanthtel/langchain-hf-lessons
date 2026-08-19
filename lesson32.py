"""lesson32 — Tools (give the LLM a real action)

GOAL (one idea only):
  Turn a Python function into something the model can call.

LangChain piece:
  @tool  -> wraps your function
  model.bind_tools([tool]) -> model may request that tool

Useful:
  - get weather
  - look up an order
  - search a database

This lesson shows the tool call request.
Running the tool + looping is the next lesson (tiny agent).

Builds on: lesson27
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage

load_dotenv()


@tool
def get_plan_price(plan_name: str) -> str:
    """Return the monthly price for a Cosmic Learning plan."""
    prices = {
        "free": "$0",
        "premium": "$9",
    }
    return prices.get(plan_name.lower(), "Unknown plan")


model = init_chat_model("ollama:llama3.2", temperature=0)

# Bind tools = "you are allowed to call these functions"
model_with_tools = model.bind_tools([get_plan_price])

response = model_with_tools.invoke(
    [HumanMessage("How much is the Premium plan?")]
)

print("Content:", response.content)
print("Tool calls:", response.tool_calls)

# If the model wants a tool, tool_calls looks like:
# [{'name': 'get_plan_price', 'args': {'plan_name': 'premium'}, ...}]
#
# Remember:
# bind_tools  → model can ASK to use a tool
# It does NOT run the tool yet (see lesson33)

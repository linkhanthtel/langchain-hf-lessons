"""lesson37 — Summarize long text (meeting notes / emails)

GOAL:
  Turn long text into a short summary + action items.

Useful in real life:
  - summarize customer emails for support agents
  - summarize meeting notes
  - summarize a help article for mobile UI

Builds on: lesson31 (structured output)
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class Summary(BaseModel):
    short_summary: str = Field(description="2-3 sentence summary")
    action_items: list[str] = Field(description="Clear next steps")
    sentiment: str = Field(description="positive | neutral | negative")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract a clear summary for a busy support manager."),
        ("human", "Customer email:\n{email}"),
    ]
)

model = init_chat_model("ollama:llama3.2", temperature=0)
chain = prompt | model.with_structured_output(Summary)

email = """
Hi team, I've been using Cosmic Learning Premium for 2 weeks.
The AI speaking feedback is great, but refunds page is confusing
and I still don't know if I can cancel before next billing date.
Please confirm the cancel steps and whether I get a partial refund.
I need an answer today because my card will be charged tomorrow.
Thanks, Maya
"""

result = chain.invoke({"email": email})

print("SUMMARY:", result.short_summary)
print("ACTIONS:")
for item in result.action_items:
    print("-", item)
print("SENTIMENT:", result.sentiment)

# Real world tip:
# Show this card next to the raw email so agents reply faster

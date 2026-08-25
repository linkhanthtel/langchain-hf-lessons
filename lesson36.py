"""lesson36 — Classify a support ticket (auto-routing)

GOAL:
  Read a customer message -> label it (billing / tech / sales).

Useful in real life:
  - send billing tickets to the billing team
  - send bug reports to engineering
  - tag emails in Zendesk / Intercom / Gmail

This uses structured output (lesson31) for a clean label.
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()


class TicketLabel(BaseModel):
    category: str = Field(description="billing | technical | sales | general")
    urgency: str = Field(description="low | medium | high")
    reason: str = Field(description="One short reason for the label")


model = init_chat_model("ollama:llama3.2", temperature=0)
classifier = model.with_structured_output(TicketLabel)

tickets = [
    "I was charged twice for Premium this month!",
    "The speaking practice page keeps crashing on iPhone.",
    "Do you have a student discount for annual Premium?",
    "Thanks for your help yesterday :)",
]

for text in tickets:
    label = classifier.invoke(
        "Classify this Cosmic Learning support message:\n" + text
    )
    print("=" * 50)
    print("MSG:", text)
    print("CATEGORY:", label.category)
    print("URGENCY:", label.urgency)
    print("REASON:", label.reason)
    print()

# Real world tip:
# After classify -> route to the right queue / Slack channel / email inbox

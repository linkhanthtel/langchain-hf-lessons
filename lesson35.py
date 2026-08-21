"""lesson35 — Chat memory (remember earlier messages)

GOAL:
  Keep a conversation going. The bot remembers what you said before.

Useful in real life:
  - website chat widget
  - WhatsApp / Telegram support bot
  - tutoring chat ("I said I'm a beginner earlier")

How it works (simple version):
  Keep a Python list called `messages`.
  Each turn: append user message → call model → append AI reply.

No LangGraph needed.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model("ollama:llama3.2", temperature=0)

# This list IS the memory for one chat session
messages = [
    SystemMessage("You are Cosmic Learning support. Keep answers short."),
]

# --- turn 1 ---
messages.append(HumanMessage("My name is Lin and my order is ORD-1001."))
reply1 = model.invoke(messages)
messages.append(AIMessage(reply1.content))
print("Bot:", reply1.content)

# --- turn 2 (no name/order repeated) ---
messages.append(HumanMessage("What was my order id again?"))
reply2 = model.invoke(messages)
messages.append(AIMessage(reply2.content))
print("Bot:", reply2.content)

# Real world tip:
# - Demo: store messages in a list (this lesson)
# - Production: save messages in a database per user/session id

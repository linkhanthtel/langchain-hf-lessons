"""lesson27 — Talk to an LLM (LangChain basics #1)

GOAL (one idea only):
  Send a message -> get a reply.

LangChain piece:
  init_chat_model(...)  -> creates a chat model
  model.invoke([...])   ->  sends messages and returns an answer

Useful in real life:
  - chatbot reply
  - rewrite an email
  - summarize a paragraph

NOT this lesson:
  - agents, tools, RAG, LangGraph  (learn those later)

Run:
  ollama serve
  python lesson27.py
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# "ollama:llama3.2" means: provider=ollama, model=llama3.2
# You could later switch to "openai:gpt-4o-mini" with almost the same code.
model = init_chat_model("ollama:llama3.2", temperature=0)

# SystemMessage = instructions for the AI's role
# HumanMessage  = what the user says
response = model.invoke(
    [
        SystemMessage("You are a friendly tutor. Keep answers under 2 sentences."),
        HumanMessage("What is LangChain in simple words?"),
    ]
)

print(response.content)

# Remember:
# response.content  → the text answer
# That is the core of almost every LangChain app.

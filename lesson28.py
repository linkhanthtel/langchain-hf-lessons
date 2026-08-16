"""lesson28 — Prompt templates (LangChain basics #2)

GOAL (one idea only):
  Reuse the same prompt with different inputs.

LangChain piece:
  ChatPromptTemplate.from_messages([...])
  prompt.invoke({...})   -> fills {variables}

Why useful:
  You don't hardcode every question.
  Same template -> many users / many topics.

Real examples:
  - "Explain {topic} for a {level} learner"
  - support reply template with {customer_name} and {issue}

Builds on: lesson27
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model("ollama:llama3.2", temperature=0)

# {topic} and {level} are placeholders filled later
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a patient teacher. Use simple words."),
        ("human", "Explain {topic} to a {level} student in 3 short bullets."),
    ]
)

# Fill the template
messages = prompt.invoke(
    {
        "topic": "embeddings",
        "level": "beginner",
    }
)

# Same as lesson27: send messages to the model
response = model.invoke(messages)
print(response.content)

# Try changing topic/level only — prompt stays the same.
# That is the point of templates.

"""lesson31 — Structured output (get JSON you can use in code)

GOAL (one idea only):
  Make the LLM return data in a fixed shape (not free text).

LangChain piece:
  model.with_structured_output(MyModel)
  -> returns a Pydantic object (topic, summary, ...)

Useful:
  - fill a database row
  - show cards in a UI
  - pass clean data to the next function

Builds on: lesson27–29
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()


class LessonSummary(BaseModel):
    title: str = Field(description="Short title")
    level: str = Field(description="beginner | intermediate | advanced")
    key_points: list[str] = Field(description="3 short bullet points")


model = init_chat_model("ollama:llama3.2", temperature=0)

# Tell the model: "answer must match this schema"
structured_model = model.with_structured_output(LessonSummary)

result = structured_model.invoke(
    "Summarize what RAG means for a beginner."
)

print(result)
print(result.title)
print(result.level)
print(result.key_points)

# Remember:
# Free text  → hard for code to use
# Structured → easy: result.title, result.key_points[0], ...

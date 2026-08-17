"""lesson29 — LCEL chain: prompt | model | parser (LangChain basics #3)

GOAL (one idea only):
  Connect steps with the pipe | operator.

LangChain piece (LCEL = LangChain Expression Language):
  chain = prompt | model | parser
  chain.invoke({...})

What each part does:
  prompt  -> builds the messages
  model   -> calls the LLM
  parser  -> turns the model output into plain text (or JSON later)

Useful:
  Cleaner than calling prompt.invoke + model.invoke by hand every time.

Builds on: lesson27 + lesson28
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Reply in one short sentence."),
        ("human", "Give one benefit of {product}."),
    ]
)

model = init_chat_model("ollama:llama3.2", temperature=0)
parser = StrOutputParser()  # takes AIMessage → plain string

# The pipe means: output of left side becomes input of right side
chain = prompt | model | parser

# One call runs the whole pipeline
answer = chain.invoke({"product": "Cosmic Learning Premium"})
print(answer)
print(type(answer))  # <class 'str'>  ← parser made it a string

# Remember:
# Without parser → you get an AIMessage object
# With StrOutputParser → you get text ready to print/store

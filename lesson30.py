"""lesson30 — Simple RAG with LangChain only (basics #4)

GOAL (one idea only):
  Find relevant text -> put it in the prompt -> ask the LLM.

This is RAG (Retrieval Augmented Generation) in the simplest form.
NO LangGraph. NO agents. Just LangChain.

Steps:
  1) store docs in a vector DB
  2) retrieve docs for a question
  3) stuff docs into a prompt
  4) LLM answers using those docs

Useful:
  - FAQ chatbot from your help articles
  - "ask my notes" apps
  - internal wiki Q&A

Builds on: lesson5 (Chroma) + lesson29 (chains)
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1) Tiny knowledge base (in real apps: many docs / PDFs) ---
docs = [
    Document(page_content="Cosmic Learning free plan needs no credit card."),
    Document(page_content="Premium costs $12/month and adds AI feedback."),
    Document(page_content="Refunds are processed within 7 business days."),
    Document(page_content="Reset password in Account Settings → Security."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 2) Prompt that includes retrieved context ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY using the context. If missing, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

model = init_chat_model("ollama:llama3.2", temperature=0)
parser = StrOutputParser()


def ask(question: str) -> str:
    # Retrieve relevant chunks
    retrieved = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in retrieved)

    # Simple chain (same idea as lesson29)
    chain = prompt | model | parser
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    questions = [
        "How much is Premium?",
        "How do I reset my password?",
        "Do you deliver pizza?",  # should say don't know
    ]
    for q in questions:
        print("=" * 50)
        print("Q:", q)
        print("A:", ask(q))
        print()

# Remember the RAG formula:
#   question → retrieve docs → prompt(context + question) → LLM answer

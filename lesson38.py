"""lesson38 — Mini FAQ bot (copy-paste starter for a real app)

GOAL:
  One small FAQ bot you could put behind a /ask API later.

Useful in real life:
  - company help center chat
  - product onboarding assistant
  - internal HR/policy Q&A

Uses only ideas you already know:
  - retriever (lesson30)
  - chain prompt | model | parser (lesson29)
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# In a real app, load these from Notion / PDF / database
FAQ_DOCS = [
    Document(page_content="Free plan: practice skills with no credit card."),
    Document(page_content="Premium costs $12/month and includes AI feedback."),
    Document(page_content="Cancel anytime in Billing Settings before renewal."),
    Document(page_content="Refunds: request within 30 days; paid in 7 business days."),
    Document(page_content="Password reset: Account Settings → Security → Reset Password."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
retriever = Chroma.from_documents(
    FAQ_DOCS, embedding=embeddings
).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Cosmic Learning FAQ bot.\n"
            "Use only this context. If unknown, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

chain = prompt | init_chat_model("ollama:llama3.2", temperature=0) | StrOutputParser()


def ask_faq(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    demo_questions = [
        "How do I cancel?",
        "What does Premium include?",
        "Can I get a refund?",
        "Where is the nearest coffee shop?",
    ]
    for q in demo_questions:
        print("=" * 50)
        print("Q:", q)
        print("A:", ask_faq(q))
        print()

# Real world next step:
# Put ask_faq() behind FastAPI: POST /ask {"question": "..."}
# (You already practiced FastAPI upload ideas in earlier lessons.)

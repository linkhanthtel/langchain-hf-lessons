"""lesson19 — Advanced RAG retrieval techniques.

Same Cosmic Learning knowledge base, three retrieval strategies:

1) similarity          — classic top-k by vector distance
2) mmr                 — diversify results (less near-duplicate chunks)
3) metadata filter     — only search billing/policy docs
4) score threshold     — drop weak matches

Goal: learn that "retrieve" is not one setting — quality depends on strategy.
No LLM needed for this lesson.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

RAW_DOCS = [
    Document(
        page_content=(
            "Free plan: practice reading, writing, listening, and speaking at no cost."
        ),
        metadata={"source": "pricing.md", "department": "billing"},
    ),
    Document(
        page_content=(
            "Premium plan costs $12/month and includes advanced AI feedback."
        ),
        metadata={"source": "pricing.md", "department": "billing"},
    ),
    Document(
        page_content=(
            "Refund window is 30 days. Approved refunds take up to 7 business days."
        ),
        metadata={"source": "refunds.md", "department": "billing"},
    ),
    Document(
        page_content=(
            "Refunds cannot be issued for accounts banned for abuse or fraud."
        ),
        metadata={"source": "refunds.md", "department": "billing"},
    ),
    Document(
        page_content=(
            "Reset password from Account Settings → Security → Reset Password."
        ),
        metadata={"source": "account.md", "department": "support"},
    ),
    Document(
        page_content=(
            "Speaking practice records your voice and shows fluency feedback."
        ),
        metadata={"source": "features.md", "department": "product"},
    ),
    Document(
        page_content=(
            "Premium also includes personalized weekly study plans and progress reports."
        ),
        metadata={"source": "pricing.md", "department": "billing"},
    ),
]


def build_vectorstore() -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
    chunks = splitter.split_documents(RAW_DOCS)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def print_docs(title: str, docs: list[Document]):
    print(f"\n=== {title} ({len(docs)} docs) ===")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. [{doc.metadata.get('department')}/{doc.metadata.get('source')}]")
        print(f"   {doc.page_content}")


if __name__ == "__main__":
    vectorstore = build_vectorstore()
    question = "Tell me about Premium pricing and refunds"

    # 1) Similarity search (default RAG)
    similarity_docs = vectorstore.similarity_search(question, k=4)
    print_docs("1) Similarity search (k=4)", similarity_docs)

    # 2) MMR — balances relevance + diversity
    mmr_docs = vectorstore.max_marginal_relevance_search(
        question,
        k=4,
        fetch_k=10,
    )
    print_docs("2) MMR search (k=4, fetch_k=10)", mmr_docs)

    # 3) Metadata filter — only billing department
    filtered_docs = vectorstore.similarity_search(
        question,
        k=4,
        filter={"department": "billing"},
    )
    print_docs("3) Metadata filter (department=billing)", filtered_docs)

    # 4) Score threshold — keep only stronger matches
    scored = vectorstore.similarity_search_with_relevance_scores(question, k=6)
    strong = [doc for doc, score in scored if score >= 0.35]
    print_docs("4) Score threshold (>= 0.35)", strong)

    print("\nTip:")
    print("- Use similarity for simple FAQs")
    print("- Use MMR when chunks are repetitive")
    print("- Use metadata filters for multi-tenant / multi-department KB")
    print("- Use score thresholds to avoid weak context in generation")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = [
    Document(
        page_content="How to use Cosmic-Learning?",
        metadata={
            "source": "user_guide.pdf",
            "department": "Learning",
            "page": 3
        }
    ),
    Document(
        page_content="Upgrading to premium features",
        metadata={
            "source": "pricing_and_billing.pdf",
            "department": "Finance",
            "page": 2
        }
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

split_docs = splitter.split_documents(docs)

for doc in split_docs:
    print(doc.page_content)
    print(doc.metadata)
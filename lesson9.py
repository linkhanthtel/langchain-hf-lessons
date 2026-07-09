from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class IngestionState(TypedDict):
    raw_text: str
    chunks: List[str]
    embeddings_count: int
    status: str


def chunk_node(state: IngestionState):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(state["raw_text"])

    return {
        "chunks": chunks,
        "status": "chunked"
    }


def embed_node(state: IngestionState):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = embeddings.embed_documents(state["chunks"])

    # In real project, store vectors in pgvector/Chroma here.
    return {
        "embeddings_count": len(vectors),
        "status": "embedded"
    }


graph = StateGraph(IngestionState)

graph.add_node("chunk", chunk_node)
graph.add_node("embed", embed_node)

graph.set_entry_point("chunk")
graph.add_edge("chunk", "embed")
graph.add_edge("embed", END)

app = graph.compile()

result = app.invoke({
    "raw_text": "Refunds take 7 business days. Employees receive 14 days leave.",
    "chunks": [],
    "embeddings_count": 0,
    "status": "uploaded"
})

print(result)
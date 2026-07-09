from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class RetrievalState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    answer: str


def retrieve_node(state: RetrievalState):
    question = state["question"]

    # Replace this with real vector DB retrieval.
    fake_chunks = [
        "Refunds are processed within 7 business days.",
        "Customers can request refunds within 30 days."
    ]

    return {
        "retrieved_chunks": fake_chunks
    }


def answer_node(state: RetrievalState):
    context = "\n".join(state["retrieved_chunks"])
    question = state["question"]

    answer = f"""
Based on the retrieved policy:

{context}

Answer:
Refunds are processed within 7 business days.
"""

    return {
        "answer": answer
    }


graph = StateGraph(RetrievalState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()

result = app.invoke({
    "question": "How long does refund take?",
    "retrieved_chunks": [],
    "answer": ""
})

print(result["answer"])
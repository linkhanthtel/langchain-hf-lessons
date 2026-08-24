from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

texts = [
    "All learners can use for free with no additional charge",
    "Learners can choose premium version to access more advanced AI features",
    "Users can reset passwords from account settings"
]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.invoke("How users can reset password?")

for doc in results:
    print(doc.page_content)

from dotenv import load_dotenv

from langchain.agents import create_agent
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

# embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    'I love React JS',
    'I enjoy FastAPI',
    'I dislike J2EE Java',
    'I am passionate about Pandas',
    'I love .NET'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('Programming', k=3))

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description='Search the most popular programming language knowledge base for information')

agent = create_agent(
    model='ollama:llama3.2',
    tools=[retriever_tool],
    system_prompt=(
        'You are a helpful assistant for questions about programming languages',
        'first call the kb_search to retrieve the context',
        'return the answer'
        )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What programming languages does the person like?"}]
})

print(result)
print(result["messages"][-1].content)


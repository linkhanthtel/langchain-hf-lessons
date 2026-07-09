from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
def get_users():
    return {"users": []}

@app.post("/users")
def create_user():
    return {"message": "created"}
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(code)

for chunk in chunks:
    print(chunk)
    print("---")
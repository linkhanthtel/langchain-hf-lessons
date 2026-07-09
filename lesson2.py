from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = """
Create Your Account. Sign up in seconds with just your email. No credit card required. Start learning immediately. Choose Your Skill
Select Reading, Writing, Listening, or Speaking. Each skill has exercises tailored to your level. Start Practicing
Begin with exercises designed for your proficiency level. Track progress and improve daily.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

chunk_vectors = embeddings.embed_documents(chunks)

print(len(chunk_vectors))
print(len(chunk_vectors[0]))
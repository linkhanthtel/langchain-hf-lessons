from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

question = "How to start practicing?"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

question_vector = embeddings.embed_query(question)

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

scores = cosine_similarity(
    [question_vector],
    chunk_vectors
)[0]

best_index = np.argmax(scores)

print("Best chunk:")
print(chunks[best_index])
print("Score:", scores[best_index])
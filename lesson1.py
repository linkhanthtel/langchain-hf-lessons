from langchain_text_splitters import RecursiveCharacterTextSplitter

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

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print("---")
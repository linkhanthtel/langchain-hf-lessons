from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text

pdf_text = extract_pdf_text("Lin_Khant_Htel.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_text(pdf_text)

print("Total chunks:", len(chunks))
print(chunks[0])
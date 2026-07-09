from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# API Documentation

## Authentication

Use Bearer tokens for authentication.

## Users API

GET /users returns all users.

## Orders API

GET /orders returns all orders.
"""

headers_to_split_on = [
    ("#", "title"),
    ("##", "section")
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = splitter.split_text(markdown_text)

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
    print("---")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
pdf_path = "data/notes.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("PDF loaded successfully!")
print("Total pages:", len(documents))
print("Total chunks:", len(chunks))

# Show first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
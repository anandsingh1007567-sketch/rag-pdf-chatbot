from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/notes.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(documents))

for i, document in enumerate(documents[:2]):
    print(f"\n--- Page {i + 1} ---")
    print(document.page_content[:500])
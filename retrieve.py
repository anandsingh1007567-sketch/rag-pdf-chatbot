from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Create embeddings
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Ask a question
question = "What is Python?"

# Retrieve relevant chunks
results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information:\n")

for i, document in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(document.page_content)
    print()
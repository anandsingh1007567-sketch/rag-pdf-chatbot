from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load PDF
loader = PyPDFLoader("data/notes.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Create embeddings using Ollama
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

# Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Embeddings created successfully!")
print("Data stored in ChromaDB!")
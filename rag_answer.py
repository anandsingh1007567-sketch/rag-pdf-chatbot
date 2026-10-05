from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# Embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# LLM
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

# User question
question = "What is Python?"

# Retrieve relevant documents
documents = retriever.invoke(question)

# Combine retrieved text
context = "\n\n".join(
    document.page_content for document in documents
)

# Prompt
prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

# Generate answer
response = llm.invoke(prompt)

print("Question:", question)
print("\nAnswer:")
print(response.content)
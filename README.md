# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) based PDF Chatbot that allows users to upload PDF documents and ask questions based on their content.

## Features

- Upload and process PDF documents
- Extract text from PDFs
- Split text into chunks
- Generate embeddings
- Store embeddings using ChromaDB
- Retrieve relevant information
- Generate answers using RAG

## Technologies Used

- Python
- LangChain
- RAG
- ChromaDB
- PDF Processing
- Generative AI

## Project Structure

- `app.py` - Main application
- `read_pdf.py` - PDF text extraction
- `split_text.py` - Text chunking
- `embed_store.py` - Embedding and vector storage
- `retrieve.py` - Relevant document retrieval
- `rag_answer.py` - RAG-based answer generation

## How to Run

```bash
pip install -r requirements.txt
python app.py

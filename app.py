import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG PDF Chatbot")
st.caption("PDF Search + Ollama AI")


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ==========================================
# MODELS
# ==========================================

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)


# ==========================================
# PDF UPLOAD
# ==========================================

st.subheader("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    # Only process when a new PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        with st.spinner("📖 Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = text_splitter.split_documents(
                documents
            )

            # Create vector database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            # Save in session
            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded_file.name

        st.success(
            f"✅ PDF loaded: {uploaded_file.name} | "
            f"Pages: {len(documents)} | "
            f"Chunks: {len(chunks)}"
        )


# ==========================================
# PDF STATUS
# ==========================================

if st.session_state.vectorstore is not None:

    st.info(
        f"📚 Searching in PDF: "
        f"{st.session_state.pdf_name}"
    )

else:

    st.warning(
        "Upload a PDF first. "
        "You can still ask Ollama general questions."
    )


# ==========================================
# CLEAR CHAT
# ==========================================

if st.button("🧹 Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ==========================================
# SHOW CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# SEARCH / QUESTION BOX
# ==========================================

question = st.chat_input(
    "🔍 Search PDF or ask Ollama..."
)


# ==========================================
# QUESTION PROCESSING
# ==========================================

if question:

    # --------------------------------------
    # Show user question
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # --------------------------------------
    # If PDF is uploaded
    # --------------------------------------

    if st.session_state.vectorstore is not None:

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching PDF..."):

                documents = retriever.invoke(question)

            context = "\n\n".join(
                document.page_content
                for document in documents
            )


            # ----------------------------------
            # First ask: Is answer in PDF?
            # ----------------------------------

            pdf_prompt = f"""
You are a PDF question answering system.

Use ONLY the PDF context below.

If the context contains enough information
to answer the question, give the answer.

If the context does NOT contain the answer,
reply with exactly:

NOT_FOUND

PDF Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("📚 Checking PDF..."):

                pdf_response = llm.invoke(
                    pdf_prompt
                )

            pdf_answer = pdf_response.content.strip()


            # ----------------------------------
            # PDF answer found
            # ----------------------------------

            if "NOT_FOUND" not in pdf_answer:

                answer = pdf_answer

                st.write(answer)

                st.success(
                    "📄 Answer found in PDF"
                )


            # ----------------------------------
            # PDF answer NOT found
            # Use Ollama general knowledge
            # ----------------------------------

            else:

                st.info(
                    "📄 Answer not found in PDF. "
                    "Asking Ollama..."
                )

                general_prompt = f"""
Answer the following question using
your general knowledge.

Question:
{question}

Give a clear and helpful answer.
"""

                with st.spinner(
                    "🤖 Ollama is generating the answer..."
                ):

                    general_response = llm.invoke(
                        general_prompt
                    )

                answer = general_response.content

                st.write(answer)

                st.success(
                    "🤖 Answer generated by Ollama"
                )


            # ----------------------------------
            # Sources
            # ----------------------------------

            with st.expander("📚 Retrieved PDF Sources"):

                for i, document in enumerate(documents):

                    page = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    if isinstance(page, int):
                        page = page + 1

                    st.write(
                        f"Source {i + 1} — Page {page}"
                    )


    # ======================================
    # NO PDF UPLOADED
    # Direct Ollama
    # ======================================

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Ollama is answering..."
            ):

                general_prompt = f"""
Answer the following question using
your general knowledge.

Question:
{question}

Give a clear and helpful answer.
"""

                response = llm.invoke(
                    general_prompt
                )

            answer = response.content

            st.write(answer)

            st.success(
                "🤖 Answer generated by Ollama"
            )


    # ======================================
    # SAVE ANSWER
    # ======================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
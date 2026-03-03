"""
RAG RETRIEVER
Builds a FAISS vector store from the clinical guidelines document.
Used to retrieve relevant clinical context before the agent answers.
This grounds the agent's responses in verified medical guidelines
rather than relying purely on Claude's training knowledge.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


KNOWLEDGE_PATH = "data/knowledge/clinical_guidelines.txt"
VECTORSTORE_PATH = "data/processed/vectorstore"


def build_vectorstore():
    """
    Reads clinical_guidelines.txt, splits into chunks,
    embeds using a local HuggingFace model (no API cost),
    saves FAISS index to disk.
    """
    if not os.path.exists(KNOWLEDGE_PATH):
        raise FileNotFoundError(f"Knowledge file not found: {KNOWLEDGE_PATH}")

    print("Building RAG vector store from clinical guidelines...")

    # Load document
    with open(KNOWLEDGE_PATH, "r") as f:
        text = f.read()

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks.")

    # Embed using local model — free, no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build and save FAISS index
    vectorstore = FAISS.from_texts(chunks, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"Vector store saved to {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore():
    """
    Loads the FAISS vector store from disk.
    Builds it first if it doesn't exist yet.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(VECTORSTORE_PATH):
        return build_vectorstore()

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_context(query: str, k: int = 3) -> str:
    """
    Given a clinical query, retrieves the top-k most relevant
    chunks from the clinical guidelines.
    Returns them as a single joined string for the prompt.
    """
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

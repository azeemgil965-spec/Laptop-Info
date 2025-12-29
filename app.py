import os
import re
import requests
import faiss
import gradio as gr
import numpy as np
import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer
from io import BytesIO

# =====================================================
# Validate API Key
# =====================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY is not set in Hugging Face Secrets.")

client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# Embedding Model
# =====================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# Knowledge Base CSV Links
# =====================================================
DOCUMENT_LINKS = [
    "https://drive.google.com/file/d/1_vAXdlwheUU15h6TNXxttmm4QgqcGR8n/view?usp=sharing"
]

# =====================================================
# Global Storage
# =====================================================
documents = []
faiss_index = None

# =====================================================
# Helpers
# =====================================================
def extract_drive_file_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def download_csv_from_drive(url):
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Invalid Google Drive link.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    return BytesIO(response.content)


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# =====================================================
# Build Knowledge Base
# =====================================================
def build_knowledge_base():
    global documents, faiss_index
    documents = []

    for link in DOCUMENT_LINKS:
        csv_file = download_csv_from_drive(link)
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            row_text = " | ".join(
                [f"{col}: {row[col]}" for col in df.columns]
            )
            documents.extend(chunk_text(row_text))

    if not documents:
        raise RuntimeError("❌ No documents loaded into the knowledge base.")

    embeddings = embedder.encode(documents, convert_to_numpy=True)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)

# Build KB at startup
build_knowledge_base()

# =====================================================
# RAG Query
# =====================================================
def ask_question(query):
    if not query.strip():
        return "### ❓ Answer\n\nPlease ask a valid question."

    if faiss_index is None:
        return "### ❌ Answer\n\nKnowledge base is not loaded."

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k=4)

    if distances[0][0] > 1.2:
        return "### ❓ Answer\n\nI don’t know. This question is outside my knowledge base."

    context = "\n\n".join([documents[i] for i in indices[0]])

    prompt = f"""
You are a strict question-answering assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return f"### ✅ Answer\n\n{response.choices[0].message.content}"

# =====================================================
# UI
# =====================================================
css = """
body {
    background: #f7f9fc;
    font-family: Inter, system-ui, sans-serif;
}
.gradio-container {
    max-width: 850px !important;
    margin: auto;
}
button {
    background-color: #6366f1 !important;
    color: white !important;
    font-weight: 600;
}
textarea {
    border-radius: 10px !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""
    # 📊 CSV Knowledge Base RAG App
    ### Answers are generated strictly from your CSV data
    """)

    question = gr.Textbox(
        label="💬 Your Question",
        placeholder="Ask a question from the CSV knowledge base...",
        lines=3
    )

    ask_btn = gr.Button("🚀 Ask Question")

    answer = gr.Markdown("### ✅ Answer\n\nYour answer will appear here.")

    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )

demo.launch()

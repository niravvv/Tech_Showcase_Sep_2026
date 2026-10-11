"""
Simple AI RAG: a RAG pipeline with DeepSeek (cloud) for answers and a free local model for embeddings.

Setup:
    pip install openai sentence-transformers numpy pypdf
    export DEEPSEEK_API_KEY=your_key_here

Usage:
    python Simple_AI_RAG.py ./docs
"""

import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LLM_MODEL = "deepseek-flash"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
embedder = SentenceTransformer(EMBED_MODEL)


#  1. Load 
def load_documents(folder):
    docs = []
    for path in Path(folder).rglob("*"):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        else:
            continue
        docs.append({"source": str(path), "text": text})
    return docs


#  2. Chunk 
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if boundary > start + size // 2:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


#  3. Embed 
def build_index(folder):
    chunks = []
    for doc in load_documents(folder):
        for c in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": c})
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    return vectors, chunks


#  4. Retrieve 
def search(question, vectors, chunks, k=TOP_K):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]


#  5. Generate 
def answer(question, vectors, chunks):
    hits = search(question, vectors, chunks)
    context = "\n\n".join(f"<doc source='{h['source']}'>\n{h['text']}\n</doc>" for h in hits)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer using only the provided documents. Cite sources by file name. "
                "If the answer isn't in the documents, say you don't know."
            )},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python Simple_AI_RAG.py <docs_folder>")
    vectors, chunks = build_index(sys.argv[1])
    print(f"Indexed {len(chunks)} chunks.")
    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in {"", "quit", "exit"}:
            break
        print("\n" + answer(q, vectors, chunks))

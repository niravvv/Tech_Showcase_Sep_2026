"""
EXERCISE 1 ANSWER: Stop re-reading everything (save and load the index)

The original Simple_AI_RAG.py re-embeds every document each time it starts.
This version saves the embeddings and chunks to a file after the first run,
then loads that file on later runs, which is much faster for large folders.

Look for the comments marked  # >>> EX1  to see exactly what changed.

Setup:
    pip install openai sentence-transformers numpy pypdf
    export DEEPSEEK_API_KEY=your_key_here

Usage:
    python ex1_answer.py ./docs      # first run: builds and saves the index
    python ex1_answer.py ./docs      # later runs: loads the saved index instantly
    python ex1_answer.py ./docs --rebuild   # force a rebuild after adding documents
"""

import os
import sys
import json                      # >>> EX1: used to turn the chunk list into text for saving
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LLM_MODEL = "deepseek-flash"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
INDEX_FILE = "rag_index.npz"     # >>> EX1: where the saved index lives

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


# >>> EX1: save the index to disk.
# np.savez stores several arrays in one file. The vectors are already a NumPy
# array; the chunks are a list of dicts, so we convert them to a JSON string first.
def save_index(vectors, chunks):
    np.savez(INDEX_FILE, vectors=vectors, chunks=json.dumps(chunks))


# >>> EX1:  load the index back from disk.
# json.loads turns the saved string back into the original list of dicts.
def load_index():
    data = np.load(INDEX_FILE)
    return data["vectors"], json.loads(str(data["chunks"]))


# >>> EX1: decide whether to load or rebuild.
# Use the saved file if it exists; otherwise build from scratch and save it.
# The --rebuild flag forces a fresh index (e.g. after adding new documents).
def get_index(folder, rebuild=False):
    if os.path.exists(INDEX_FILE) and not rebuild:
        print(f"Loading saved index from {INDEX_FILE}...")
        return load_index()
    print("Building index (this only happens once)...")
    vectors, chunks = build_index(folder)
    save_index(vectors, chunks)
    print(f"Saved index to {INDEX_FILE}")
    return vectors, chunks


#   4. Retrieve  
def search(question, vectors, chunks, k=TOP_K):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]


#   5. Generate  
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
        sys.exit("Usage: python ex1_answer.py <docs_folder> [--rebuild]")

    # >>> EX1: CHANGED get_index() replaces the direct call to build_index()
    vectors, chunks = get_index(sys.argv[1], rebuild="--rebuild" in sys.argv)
    print(f"Ready: {len(chunks)} chunks.")

    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in {"", "quit", "exit"}:
            break
        print("\n" + answer(q, vectors, chunks))

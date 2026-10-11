"""
EXERCISE 2 ANSWER: Show your sources, and know when to say "I don't know"

The original Simple_AI_RAG.py always sends the top 4 chunks to DeepSeek, even
when none of them is relevant. This version:
  1. prints each retrieved chunk's file name and similarity score, and
  2. skips the API call entirely if even the best match is below a threshold.

Look for the comments marked  # >>> EX2  to see exactly what changed.

Setup:
    pip install openai sentence-transformers numpy pypdf
    export DEEPSEEK_API_KEY=your_key_here

Usage:
    python ex2_answer.py ./docs
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

# >>> EX2: the minimum similarity score needed to count as "relevant".
# Cosine similarity runs from -1 to 1; with all-MiniLM-L6-v2, scores below ~0.3
# are usually unrelated. Tune it: ask questions you know ARE and AREN'T covered
# by your documents, watch the printed scores, and pick a value in between.
MIN_SCORE = 0.3

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
    # >>> EX2: the original returned just chunks[i] and threw the
    # scores away. Now we copy each chunk and attach its similarity score.
    # (np.argsort sorts best-first here, so hits[0] is always the top match.)
    return [{**chunks[i], "score": float(scores[i])} for i in top]


#  5. Generate 
def answer(question, vectors, chunks):
    hits = search(question, vectors, chunks)

    # >>> EX2: show the user which files were retrieved and how closely
    # each one matched, so they can see WHY the answer says what it says.
    print("\nSources retrieved:")
    for h in hits:
        print(f"  {h['score']:.2f}  {h['source']}")

    # >>> EX2: the relevance check. If the best match is too weak, nothing
    # in the documents is about this question, so don't call DeepSeek at all.
    # This saves money (no API call) and prevents answers built on junk context.
    if hits[0]["score"] < MIN_SCORE:
        return "I couldn't find that in your documents."

    # >>> EX2: OPTIONAL also drop any individual chunks below the threshold,
    # so only genuinely relevant passages are sent to the model.
    hits = [h for h in hits if h["score"] >= MIN_SCORE]

    # Everything below is unchanged from Simple_AI_RAG.py
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
        sys.exit("Usage: python ex2_answer.py <docs_folder>")
    vectors, chunks = build_index(sys.argv[1])
    print(f"Indexed {len(chunks)} chunks.")
    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in {"", "quit", "exit"}:
            break
        print("\n" + answer(q, vectors, chunks))

"""
EXERCISE 3 ANSWER: Go fully local and private with Ollama

The original Simple_AI_RAG.py sends your question and document chunks to
DeepSeek's cloud API. This version sends them to a model running on your own
computer through Ollama, so no API key is needed and nothing leaves your machine.

Because Ollama offers an OpenAI-compatible API (just like DeepSeek), the only
real changes are the client setup and the model name.

Look for the comments marked  # >>> EX3  to see exactly what changed.

Setup:
    1. Install Ollama from https://ollama.com/download (it starts automatically)
    2. ollama pull qwen3:8b
         - about 16 GB RAM: qwen3:8b
         - lighter laptop:  qwen3:4b
         - 32 GB+ or a GPU: qwen3:14b
    3. pip install openai sentence-transformers numpy pypdf
       (no DEEPSEEK_API_KEY needed any more)

Usage:
    python ex3_answer.py ./docs
"""

import sys                       # >>> EX3: "os" is no longer needed - there's no API key to read
from pathlib import Path

import numpy as np
from openai import OpenAI        # same library as before: Ollama speaks the OpenAI format too
from sentence_transformers import SentenceTransformer

# >>> EX3: a local model name instead of "deepseek-flash".
# It must match a model you've downloaded with "ollama pull" (check with "ollama list").
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "all-MiniLM-L6-v2"   # unchanged: embeddings already ran locally
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

# >>> EX3: point the client at Ollama on your own computer.
#   base_url: Ollama's OpenAI-compatible server (localhost = this machine)
#   api_key:  the library requires a value, but Ollama doesn't check it
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
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
        model=LLM_MODEL,     # >>> EX3: now refers to the local Ollama model
        messages=[
            {"role": "system", "content": (
                "Answer using only the provided documents. Cite sources by file name. "
                "If the answer isn't in the documents, say you don't know."
                # >>> EX3: Qwen3 can "think out loud" before answering, which
                # is slow for simple document Q&A. "/no_think" switches that off.
                " /no_think"
            )},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
    )
    reply = response.choices[0].message.content
    # >>> EX3: some Ollama versions still include an empty <think></think>
    # block at the start of the reply. Strip it so only the answer is shown.
    if "</think>" in reply:
        reply = reply.split("</think>", 1)[1]
    return reply.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python ex3_answer.py <docs_folder>")
    vectors, chunks = build_index(sys.argv[1])
    print(f"Indexed {len(chunks)} chunks. Running fully offline with {LLM_MODEL}.")
    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in {"", "quit", "exit"}:
            break
        # >>> EX3: if this fails with "Connection refused", Ollama isn't
        # running. Open the Ollama app, or run "ollama serve" in another terminal.
        print("\n" + answer(q, vectors, chunks))

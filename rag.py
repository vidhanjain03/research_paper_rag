import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ── CONFIG ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GEMINI_API_KEY")   # from console.groq.com
PAPERS_FILE  = "researchpaper.txt"
TOP_K        = 5
MODEL_ID     = "llama-3.1-8b-instant"
# ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful research assistant who has read several research papers.
Your job is to answer questions about these papers in a friendly, conversational way.

Rules:
- Answer based on the provided context
- If asked general questions like "tell me anything" or "how are you", respond naturally
- If asked about the papers generally, summarize what you know
- If a specific detail isn't in the context, say so but still be helpful with what you do know
- Be concise but informative
"""

def load_and_chunk(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()

    if not text.strip():
        print(f"❌ {filepath} is empty!")
        return []

    # Split the text using the exact 60-dash separator from the extractor
    raw_papers = text.split("-" * 60)
    
    chunks = []
    for paper in raw_papers:
        cleaned_paper = paper.strip()
        
        # Only append if the chunk contains actual text
        if cleaned_paper:
            chunks.append(cleaned_paper)
            
    return chunks

def build_index(chunks, embedder):
    print("Building index...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, embedder, top_k=TOP_K):
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    faiss.normalize_L2(query_vec)
    _, indices = index.search(query_vec, top_k)
    
    # Ensure we don't look up out-of-bounds indices if top_k > len(chunks)
    valid_indices = [i for i in indices[0] if i < len(chunks)]
    return [chunks[i] for i in valid_indices]

def answer(client, query, context_chunks, chat_history):
    context = "\n\n".join(context_chunks)

    # Build messages: system + history + new user message with context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history
    messages.append({
        "role": "user",
        "content": f"""Context from research papers:
{context}

User question: {query}"""
    })

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=1024,
        temperature=0.4
    )
    return response.choices[0].message.content

def main():
    if not os.path.exists(PAPERS_FILE):
        print(f"❌ '{PAPERS_FILE}' not found in {os.getcwd()}")
        return

    client   = Groq(api_key=GROQ_API_KEY)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = load_and_chunk(PAPERS_FILE)
    if not chunks:
        print("❌ No content loaded.")
        return

    print(f"Loaded {len(chunks)} chunks (individual papers) from your database.")
    index = build_index(chunks, embedder)

    print("\n✅ Ready! Ask questions about your research papers.")
    print("Type 'exit' to quit.\n")

    chat_history = []   # keeps conversation memory

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        # Adjust top_k dynamically if total papers are fewer than default TOP_K
        current_top_k = min(TOP_K, len(chunks))
        relevant  = retrieve(query, index, chunks, embedder, top_k=current_top_k)
        response  = answer(client, query, relevant, chat_history)

        # Update history (keep last 6 turns to avoid token overflow)
        chat_history.append({"role": "user",      "content": query})
        chat_history.append({"role": "assistant", "content": response})
        chat_history = chat_history[-12:]

        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()
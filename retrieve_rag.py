import os 
# Thread control - MUST be before any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import sqlite3
import time
import multiprocessing 
import numpy as np
from google import genai 

# Add this right after your imports, BEFORE loading SentenceTransformer
if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)

from sentence_transformers import SentenceTransformer


# Ensure these don't run heavy initialization on import-time if possible
try:
    from ingest import VectorStore, DB_PATH, FAISS_INDEX_PATH, EMBED_MODEL
except ImportError:
    print("Error: Could not import from 'ingest.py'. Check your file structure.")
    sys.exit(1)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# Initialize Gemini
# DATA PRIVACY: Never paste the key directly in the code.
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
    client = None
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        client = None

RAG_MODEL = "gemini-2.0-flash" # Updated to latest flash model version if available, or keep 1.5
MODEL_NAME = EMBED_MODEL 

print("Loading Embedding Model... (this happens once)")
# Loading this globally is fine for a CLI script, but usually bad for a web server
_GLOBAL_MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
print("Model Loaded.")

# -------------------------------------------------------
# Database
# -------------------------------------------------------
def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------------------------------
# Step 1 — Retrieve 
# -------------------------------------------------------
def retrieve(query, top_k=5):
    # 1. Encode query
    # This is where your Segfault was happening previously
    qvec = _GLOBAL_MODEL.encode([query], normalize_embeddings=True).astype("float32")

    # 2. Vector Search
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index not found at {FAISS_INDEX_PATH}")
        return []
        
    vs = VectorStore(index_path=FAISS_INDEX_PATH, dim=None)
    results = vs.search(qvec, k=top_k)

    if not results:
        return []

    chunk_ids = [r[0] for r in results]
    scores = {r[0]: r[1] for r in results}

    # 3. Fetch Metadata from SQLite
    conn = get_db_conn()
    cur = conn.cursor()

    placeholders = ",".join("?" * len(chunk_ids))
    sql = f"""
        SELECT 
            c.id as chunk_id, c.text, c.chunk_index, c.page_number,
            d.id as doc_id, d.title, d.filename, d.manual_relevance
        FROM chunks c 
        JOIN documents d ON d.id = c.document_id 
        WHERE c.id IN ({placeholders})
    """
    cur.execute(sql, tuple(chunk_ids))
    rows = cur.fetchall()
    conn.close()

    # 4. Format Results
    retrieved = []
    for row in rows:
        cid = row["chunk_id"]
        retrieved.append({
            "doc_id": row["doc_id"],
            "title": row["title"],
            "filename": row["filename"],
            "chunk_index": row["chunk_index"],
            "page": row["page_number"],
            "text": clean_text(row["text"]),
            "score": float(scores.get(cid, 0.0)),
            "manual_stars": row["manual_relevance"],
        })

    retrieved.sort(key=lambda x: x["score"], reverse=True)
    return retrieved

def clean_text(t: str) -> str:
    if not t: return ""
    t = t.replace("-\n", "")
    t = t.replace("\n", " ")
    return " ".join(t.split()) # More robust whitespace cleaning

# -------------------------------------------------------
# Step 2 — Generate
# -------------------------------------------------------
def generate_rag_answer(query: str, retrieved_chunks: list, model=RAG_MODEL):
    if not client:
        return "Gemini client is not initialized. Check API Key."
    
    if not retrieved_chunks:
        return "No relevant information found in your knowledge base."

    context = ""
    for c in retrieved_chunks:
        context += (
            f"[Source: {c['filename']}, page {c.get('page', '?')}] \n"
            f"{c['text']}\n\n---\n\n"
        )

    prompt = f"""
You are an expert assistant. The user asked: "{query}"

Context:
{context}

Based ONLY on the context above:
1. Explain clearly and concisely.
2. Cite sources (filename/page) at the end.
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.3},
        )
        return response.text
    except Exception as e:
        return f"An API error occurred: {e}"

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    # Check Python version env
    print(f"Running on Python {sys.version.split()[0]}")
    
    while True:
        try:
            q = input("\n🔎 Query (or 'exit'): ")
            if q.strip().lower() == "exit":
                break
            
            answer, chunks = retrieve(q, top_k=4), [] # Split logic for debugging
            
            # Run generation only if retrieval worked
            if answer:
                 final_ans = generate_rag_answer(q, answer)
            else:
                 final_ans = "Nothing found."

            print("\n============================")
            print("Final Answer")
            print("============================\n")
            print(final_ans)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n An error occurred: {e}")
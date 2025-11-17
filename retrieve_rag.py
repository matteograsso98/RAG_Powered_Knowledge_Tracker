import sqlite3
import time
from sentence_transformers import SentenceTransformer
from ingest import VectorStore, DB_PATH

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------------------------------
# GLOBAL MODEL LOADING
# Loading here ensures it only happens once when app starts,
# not on every single search request.
# -------------------------------------------------------
print("Loading Embedding Model... (this happens once)")
_GLOBAL_MODEL = SentenceTransformer(MODEL_NAME)
print("Model Loaded.")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name: row['title']
    return conn


def retrieve(query, top_k=5):
    """
    1. Encode query using global model.
    2. Search FAISS vector store.
    3. Fetch corresponding text from SQLite.
    """
    t_start = time.time()
    
    # 1. Encode
    qvec = _GLOBAL_MODEL.encode([query], convert_to_numpy=True)
    
    # 2. Vector Search
    # Note: We initialize VectorStore here, which loads the index from disk.
    # For extreme low-latency, you could also load the VectorStore globally,
    # but loading the index is usually fast enough (ms) compared to the model load (sec).
    vs = VectorStore()
    results = vs.search(qvec, k=top_k) 
    # results = list of (chunk_id, score)
    
    if not results:
        return []

    chunk_ids = [r[0] for r in results]
    scores = {r[0]: r[1] for r in results} # map id -> score

    # 3. SQL Fetch (Optimized: Single Query)
    conn = get_db_conn()
    cur = conn.cursor()
    
    placeholders = ",".join("?" * len(chunk_ids))
    sql = f"""
        SELECT 
            c.id as chunk_id, 
            c.text, 
            c.chunk_index, 
            d.id as doc_id, 
            d.title, 
            d.filename,
            d.manual_relevance
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
        cid = row['chunk_id']
        # Calculate a weighted score? 
        # e.g., Vector Score + (Manual Stars * 0.1)
        # For now, just returning raw vector score.
        vec_score = float(scores.get(cid, 0.0))
        
        retrieved.append({
            "doc_id": row['doc_id'],
            "title": row['title'],
            "filename": row['filename'],
            "chunk_index": row['chunk_index'],
            "text": row['text'],
            "score": vec_score,
            "manual_stars": row['manual_relevance']
        })
    
    # Re-sort because SQL might return in arbitrary order
    retrieved.sort(key=lambda x: x['score'], reverse=True)
    
    # Debug timing
    # print(f"Search took {time.time() - t_start:.4f}s")
    
    return retrieved

if __name__ == "__main__":
    # CLI Test
    while True:
        q = input("\n🔎 Query (or 'exit'): ")
        if q.strip().lower() == 'exit': break
        
        hits = retrieve(q, top_k=3)
        for h in hits:
            print(f"\n {h['title']} (Score: {h['score']:.3f})")
            print(f"   {h['text'][:200]}...")

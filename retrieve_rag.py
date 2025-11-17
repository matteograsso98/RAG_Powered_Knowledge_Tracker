# retrieve_rag.py
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import VectorStore, DB_PATH  # reuse

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

def get_db_conn():
    return sqlite3.connect(DB_PATH)

def retrieve(query, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode([query], convert_to_numpy=True)
    vs = VectorStore(dim=model.get_sentence_embedding_dimension())
    results = vs.search(qvec, top_k)
    # results: list of (chunk_id, score)
    conn = get_db_conn()
    cur = conn.cursor()
    retrieved = []
    for chunk_id, score in results:
        cur.execute("SELECT c.text, c.chunk_index, d.title, d.filename, d.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ?", (chunk_id,))
        row = cur.fetchone()
        if row:
            text, chunk_idx, title, filename, doc_id = row
            retrieved.append({
                "doc_id": doc_id,
                "title": title,
                "filename": filename,
                "chunk_index": chunk_idx,
                "text": text,
                "score": score
            })
    conn.close()
    return retrieved

if __name__ == "__main__":
    q = input("Query: ")
    hits = retrieve(q, top_k=6)
    for h in hits:
        print("----")
        print(f"{h['title']} (score {h['score']:.3f})")
        print(h['text'][:500].strip())
        print("source:", h['filename'], "chunk", h['chunk_index'])

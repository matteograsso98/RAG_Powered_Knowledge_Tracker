import os
# --- THREAD CONTROL (MUST BE FIRST) ---
# Prevents Segmentation Fault 11 on macOS by forcing single-threaded operation for libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sqlite3
import hashlib
import json
import time
import faiss
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader

# Add this right after your imports, BEFORE loading SentenceTransformer
if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)

from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_PATH = "rag_library.db"
DATA_ROOT = "data/docs"
FAISS_INDEX_PATH = "data/faiss_index.bin"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# -------------------------------------------------------
# VectorStore Class (IndexFlatIP + IDMap)
# -------------------------------------------------------
class VectorStore:
    def __init__(self, index_path=FAISS_INDEX_PATH, dim=None, model: SentenceTransformer = None):
        self.index_path = index_path
        self.dim = dim
        self.index = None
        self.model = model
        self.load()

    def _create_index(self, dim):
        base_index = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIDMap(base_index)
        return idx

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d
        else:
            if self.dim is None:
                self.index = None
            else:
                self.index = self._create_index(self.dim)

    def save(self):
        if self.index is None:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def add(self, vectors: np.ndarray, ids: list):
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.dtype != np.float32:
            vectors = vectors.astype('float32')

        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = self._create_index(self.dim)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vectors = vectors / norms

        ids_np = np.array(ids).astype('int64')
        self.index.add_with_ids(vectors, ids_np)

    def remove(self, ids: list):
        if not ids:
            return
        ids_np = np.array(ids).astype('int64')
        try:
            self.index.remove_ids(ids_np)
        except Exception as e:
            print("Warning: remove_ids failed, falling back to rebuild-index approach:", e)
            self._rebuild_excluding(ids_np.tolist())

    def _rebuild_excluding(self, exclude_ids: list):
        print("Rebuilding FAISS index excluding", len(exclude_ids), "ids (this may be slow)...")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, text FROM chunks WHERE id NOT IN (%s)" %
                    ",".join("?" * len(exclude_ids)), tuple(exclude_ids))
        rows = cur.fetchall()
        conn.close()

        vecs = []
        ids = []
        if not rows:
            self.index = self._create_index(self.dim or 1)
            return

        if self.model is None:
            model = SentenceTransformer(EMBED_MODEL)
        else:
            model = self.model

        for r in rows:
            cid, txt = r
            emb = model.encode(txt, normalize_embeddings=True).astype('float32')
            vecs.append(emb)
            ids.append(int(cid))

        self.index = self._create_index(vecs[0].shape[0])
        self.add(np.vstack(vecs), ids)
        conn.close()
        print("Rebuild complete.")

    def search(self, query_vector: np.ndarray, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.asarray(query_vector)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.dtype != np.float32:
            q = q.astype('float32')
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        q = q / norm
        distances, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            results.append((int(idx), float(score)))
        return results

# -------------------------------------------------------
# Text Processing Utilities
# -------------------------------------------------------
def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    if not text:
        return []
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = max(end - overlap, end) if end - overlap > start else end
    return chunks

def extract_text(path: Path):
    ext = path.suffix.lower()
    # Returns list of (page_number, text)
    if ext == ".pdf":
        pages = []
        try:
            reader = PdfReader(path)
            for i, p in enumerate(reader.pages):
                txt = p.extract_text() or ""
                pages.append((i + 1, clean_text(txt)))
            return pages
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return []
    elif ext in [".txt", ".md"]:
        try:
            txt = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            return [(None, txt)]
        except Exception as e:
            print(f"Error reading Text file {path}: {e}")
            return []
    else:
        print(f"Skipping unsupported file: {path}")
        return []

# -------------------------------------------------------
# DB Categories
# -------------------------------------------------------
def get_or_create_category(conn, path_parts):
    cur = conn.cursor()
    parent_id = None
    for part in path_parts:
        cur.execute(
            "SELECT id FROM categories WHERE name = ? AND parent_id IS ?",
            (part, parent_id)
        )
        row = cur.fetchone()
        if row:
            cat_id = row[0]
        else:
            cur.execute(
                "INSERT INTO categories (name, parent_id) VALUES (?, ?)",
                (part, parent_id)
            )
            cat_id = cur.lastrowid
        parent_id = cat_id
    conn.commit()
    return parent_id

def cleanup_empty_categories(conn):
    cur = conn.cursor()
    print("Cleaning up empty categories...")
    changes_made = True
    while changes_made:
        changes_made = False
        cur.execute("""
            SELECT c.id 
            FROM categories c
            LEFT JOIN documents d ON d.category_id = c.id
            WHERE c.id NOT IN (SELECT DISTINCT parent_id FROM categories WHERE parent_id IS NOT NULL)
            AND d.id IS NULL
        """)
        ids_to_delete = [row[0] for row in cur.fetchall()]
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cur.execute(f"DELETE FROM categories WHERE id IN ({placeholders})", ids_to_delete)
            conn.commit()
            print(f"   Deleted {len(ids_to_delete)} empty categories.")
            changes_made = True
    if not changes_made:
        print("   Category cleanup finished.")

# -------------------------------------------------------
# Main Ingestion Logic
# -------------------------------------------------------
def ingest_documents():
    print("ingestion starting…")
    
    # Load Model
    model = SentenceTransformer(EMBED_MODEL)
    embed_dim = model.get_sentence_embedding_dimension()
    
    # Initialize Vector Store
    vs = VectorStore(dim=embed_dim, model=model)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Schema
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        parent_id INTEGER,
        UNIQUE(name, parent_id)
    );
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        filename TEXT,
        file_path TEXT,
        category_id INTEGER,
        manual_relevance INTEGER DEFAULT 0,
        auto_relevance REAL DEFAULT 0,
        access_count INTEGER DEFAULT 0,
        file_hash TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(category_id) REFERENCES categories(id)
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_index INTEGER,
        page_number INTEGER,
        text TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        event_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    for root, dirs, files in os.walk(DATA_ROOT):
        root_path = Path(root)
        try:
            rel_path = root_path.relative_to(DATA_ROOT)
        except ValueError:
            rel_path = Path(".")

        category_parts = [] if str(rel_path) == "." else list(rel_path.parts)
        category_id = get_or_create_category(conn, category_parts) if category_parts else None

        for fname in files:
            full_path = root_path / fname
            if full_path.name.startswith('.'): continue

            file_hash = hash_file(full_path)

            cur.execute("SELECT id, file_hash, category_id FROM documents WHERE filename = ?", (fname,))
            row = cur.fetchone()

            if row:
                doc_id, old_hash, old_category_id = row

                # UPDATE LOGIC
                if old_hash != file_hash:
                    print(f"♻️ Updating changed file: {fname}")
                    cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
                    old_chunk_ids = [r[0] for r in cur.fetchall()]
                    if old_chunk_ids:
                        try:
                            vs.remove(old_chunk_ids)
                        except Exception as e:
                            print("Warning removing old vectors:", e)
                        placeholders = ",".join("?" * len(old_chunk_ids))
                        cur.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", tuple(old_chunk_ids))

                    pages_data = extract_text(full_path)
                    if not pages_data:
                        cur.execute("UPDATE documents SET file_hash=?, file_path=?, category_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                    (file_hash, str(full_path), category_id, doc_id))
                        conn.commit()
                        continue

                    new_ids, new_vecs = [], []
                    chunk_index_counter = 0

                    for page_num, page_text in pages_data:
                        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                        for chunk in page_chunks:
                            cur.execute(
                                "INSERT INTO chunks (document_id, chunk_index, page_number, text) VALUES (?, ?, ?, ?)", 
                                (doc_id, chunk_index_counter, page_num, chunk)
                            )
                            cid = cur.lastrowid
                            new_ids.append(cid)
                            new_vecs.append(model.encode(chunk, normalize_embeddings=True).astype('float32'))
                            chunk_index_counter += 1

                    if new_ids:
                        vs.add(np.vstack(new_vecs), new_ids)

                    cur.execute("UPDATE documents SET file_hash=?, file_path=?, category_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (file_hash, str(full_path), category_id, doc_id))
                    conn.commit()

                elif old_category_id != category_id:
                    print(f"🔄 Document category updated: {fname}")
                    cur.execute("UPDATE documents SET category_id=?, file_path=?, file_hash=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                                (category_id, str(full_path), file_hash, doc_id))
                    conn.commit()

            else:
                # NEW FILE LOGIC
                print(f"➕ Ingesting new: {fname}")
                pages_data = extract_text(full_path)
                if not pages_data: continue

                cur.execute(
                    "INSERT INTO documents (title, filename, file_path, category_id, file_hash) VALUES (?, ?, ?, ?, ?)",
                    (fname, fname, str(full_path), category_id, file_hash)
                )
                doc_id = cur.lastrowid
                
                new_ids, new_vecs = [], []
                chunk_index_counter = 0

                for page_num, page_text in pages_data:
                    page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                    for chunk in page_chunks:
                        cur.execute(
                            "INSERT INTO chunks (document_id, chunk_index, page_number, text) VALUES (?, ?, ?, ?)", 
                            (doc_id, chunk_index_counter, page_num, chunk)
                        )
                        cid = cur.lastrowid
                        new_ids.append(cid)
                        new_vecs.append(model.encode(chunk, normalize_embeddings=True).astype('float32'))
                        chunk_index_counter += 1
                
                if new_ids:
                    vs.add(np.vstack(new_vecs), new_ids)
                
                conn.commit()

    vs.save()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    t0 = time.time()
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        print(f"Created {DATA_ROOT}. Put your PDFs there!")
    ingest_documents()
    conn = sqlite3.connect(DB_PATH)
    cleanup_empty_categories(conn)
    conn.close()
    print(f"Done in {time.time() - t0:.2f} seconds")
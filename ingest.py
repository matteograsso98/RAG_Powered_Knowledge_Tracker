import os
import sqlite3
import hashlib
import json
import time
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# --- Configuration ---
DB_PATH = "rag_library.db"
DATA_ROOT = "data/docs"
FAISS_INDEX_PATH = "data/faiss_index.bin"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# -------------------------------------------------------
# VectorStore Class (Fixes ImportError and ID alignment)
# -------------------------------------------------------
class VectorStore:
    def __init__(self, index_path=FAISS_INDEX_PATH, dim=384):
        self.index_path = index_path
        self.dim = dim
        self.index = None
        self.load()

    def load(self):
        """Load index from disk or create a new IDMap index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # IndexIDMap allows us to supply specific IDs (chunk_ids) 
            # rather than having FAISS auto-increment them.
            base_index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(base_index)

    def add(self, vectors, ids):
        """
        vectors: np.array of shape (n, dim)
        ids: list of integers (chunk_ids from DB)
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype('float32')
        
        ids_np = np.array(ids).astype('int64')
        self.index.add_with_ids(vectors, ids_np)

    def remove(self, ids):
        """Remove vectors by their specific Chunk IDs."""
        if not ids:
            return
        ids_np = np.array(ids).astype('int64')
        self.index.remove_ids(ids_np)

    def save(self):
        """Save index to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def search(self, query_vector, k=5):
        """Returns list of (chunk_id, score)"""
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        # indices[0] contains the IDs we passed in add_with_ids
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))
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

def chunk_text(text, size=1200, overlap=200):
    chunks = []
    start = 0
    if not text: return []
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def extract_text(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            text = ""
            for p in reader.pages:
                text += p.extract_text() or ""
            return clean_text(text)
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return ""
    elif ext in [".txt", ".md"]:
        try:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
             print(f"Error reading Text file {path}: {e}")
             return ""
    else:
        print(f"Skipping unsupported file: {path}")
        return ""

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
    """
    Recursively removes categories that have no associated documents 
    and no child categories.
    """
    cur = conn.cursor()
    print("🧹 Cleaning up empty categories...")
    changes_made = True
    
    # Loop until no more empty categories are found (ensures nested subfolders are cleaned)
    while changes_made:
        changes_made = False
        
        # Select category IDs that are empty: 
        # 1. No child categories AND 
        # 2. No associated documents
        cur.execute("""
            SELECT c.id 
            FROM categories c
            LEFT JOIN documents d ON d.category_id = c.id
            WHERE c.id NOT IN (SELECT DISTINCT parent_id FROM categories WHERE parent_id IS NOT NULL)
            AND d.id IS NULL
        """)
        
        ids_to_delete = [row[0] for row in cur.fetchall()]
        
        if ids_to_delete:
            # Delete categories
            placeholders = ",".join("?" * len(ids_to_delete))
            cur.execute(f"DELETE FROM categories WHERE id IN ({placeholders})", ids_to_delete)
            conn.commit()
            print(f"   Deleted {len(ids_to_delete)} empty categories.")
            changes_made = True
        
    if not changes_made:
        print("   Category cleanup finished (no empty categories remaining).")

# -------------------------------------------------------
# Main Ingestion Logic
# -------------------------------------------------------
def ingest_documents():
    print("ingestion starting…")
    
    # Load Model
    model = SentenceTransformer(EMBED_MODEL)
    embed_dim = model.get_sentence_embedding_dimension()
    
    # Initialize Vector Store
    vs = VectorStore(dim=embed_dim)

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

        # Create Categories
        category_parts = [] if str(rel_path) == "." else list(rel_path.parts)
        category_id = get_or_create_category(conn, category_parts) if category_parts else None

        for fname in files:
            full_path = root_path / fname
            if full_path.name.startswith('.'): continue # skip hidden files
            
            file_hash = hash_file(full_path)
            
            # Check DB
            cur.execute("SELECT id, file_hash, category_id FROM documents WHERE filename = ?", (fname,))
            row = cur.fetchone()

            if row:
                doc_id, old_hash, old_category_id = row
                
                # Check 1: Has the file content changed? (Standard RAG update)
                if old_hash != file_hash:
                    # Logic to remove old chunks, re-chunk, re-embed, and update hash
                    # (Your existing re-ingestion logic goes here)
                    # ...
                    pass # Ensure you update the category_id here as well
                
                # Check 2: Has the file location (category) changed? (Folder structure update)
                if old_category_id != category_id:
                    print(f"🔄 Document category updated: {fname}")
                    cur.execute("UPDATE documents SET category_id=?, file_path=?, file_hash=? WHERE id=?", 
                                (category_id, str(full_path), file_hash, doc_id))
                    conn.commit()
                    # We don't need to re-embed if the hash is the same, just update metadata.
                
            else:
                # New File (Original ingestion logic)
                print(f"➕ Ingesting new: {fname}")
                text = extract_text(full_path)
                if not text: continue

                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                
                cur.execute(
                    "INSERT INTO documents (title, filename, file_path, category_id, file_hash) VALUES (?, ?, ?, ?, ?)",
                    (fname, fname, str(full_path), category_id, file_hash)
                )
                doc_id = cur.lastrowid
                
                new_ids = []
                new_vecs = []
                
                for idx, chunk in enumerate(chunks):
                    cur.execute("INSERT INTO chunks (document_id, chunk_index, text) VALUES (?, ?, ?)", (doc_id, idx, chunk))
                    cid = cur.lastrowid
                    new_ids.append(cid)
                    new_vecs.append(model.encode(chunk))
                
                if new_ids:
                    vs.add(np.array(new_vecs), new_ids)
                
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
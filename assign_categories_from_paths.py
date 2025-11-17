# assign_categories_from_paths.py
import sqlite3
from pathlib import Path
from utils.categories import ensure_category_path

DB_PATH = "rag_library.db"
DOCS_ROOT = Path("data/docs")  # adjust if different

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename FROM documents")
    rows = cur.fetchall()
    updated = 0
    for doc_id, filename in rows:
        try:
            p = Path(filename)
            # if filename is absolute or different root, try to make relative to DOCS_ROOT
            try:
                rel = p.relative_to(DOCS_ROOT)
            except Exception:
                # fallback: search for DOCS_ROOT in path string
                parts = p.parts
                if "data" in parts and "docs" in parts:
                    idx = parts.index("docs")
                    rel = Path(*parts[idx+1:])
                else:
                    rel = None
            if rel is None:
                continue
            dir_parts = rel.parts[:-1]  # exclude file name
            if not dir_parts:
                continue
            category_id = ensure_category_path(conn, list(dir_parts))
            cur.execute("UPDATE documents SET category_id = ? WHERE id = ?", (category_id, doc_id))
            updated += 1
        except Exception as e:
            print("error", doc_id, e)
    conn.commit()
    conn.close()
    print("Done. Updated", updated, "documents.")

if __name__ == "__main__":
    main()
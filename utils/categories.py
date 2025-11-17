# utils/categories.py
import sqlite3

def get_or_create_category(conn: sqlite3.Connection, name: str, parent_id):
    """
    Returns category_id for name under parent_id.
    If not exists, creates it.
    """
    cur = conn.cursor()
    if parent_id is None:
        cur.execute("SELECT id FROM categories WHERE name = ? AND parent_id IS NULL", (name,))
    else:
        cur.execute("SELECT id FROM categories WHERE name = ? AND parent_id = ?", (name, parent_id))
    row = cur.fetchone()
    if row:
        return row[0]
    # create
    cur.execute("INSERT INTO categories (parent_id, name) VALUES (?, ?)", (parent_id, name))
    conn.commit()
    return cur.lastrowid

def ensure_category_path(conn: sqlite3.Connection, path_parts):
    """
    path_parts: list of folder names, e.g. ["AI","LanguageModels"]
    Returns deepest category_id (or None if path_parts is empty)
    """
    parent = None
    for p in path_parts:
        parent = get_or_create_category(conn, p, parent)
    return parent
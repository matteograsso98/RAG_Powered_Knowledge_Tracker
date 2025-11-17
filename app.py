# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import sqlite3
from retrieve_rag import retrieve, get_db_conn
from utils.categories import get_or_create_category

app = Flask(__name__)
DB = "rag_library.db"

def db_conn():
    return sqlite3.connect(DB)

# helper to fetch category tree
def fetch_categories(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, parent_id, name FROM categories ORDER BY parent_id NULLS FIRST, name")
    rows = cur.fetchall()
    # build simple nested dict
    tree = {}
    by_id = {}
    for r in rows:
        cid, parent, name = r
        by_id[cid] = {"id": cid, "parent": parent, "name": name, "children": []}
    for cid, node in by_id.items():
        if node["parent"] and node["parent"] in by_id:
            by_id[node["parent"]]["children"].append(node)
        else:
            tree[cid] = node
    return list(tree.values())

@app.route("/")
def index():
    conn = db_conn()
    cur = conn.cursor()
    category_id = request.args.get("category_id", type=int)

    sql = "SELECT id, title, filename, manual_relevance, auto_relevance, access_count FROM documents "
    params = []
    
    if category_id is not None:
        # Find all descendants of the current category (including itself)
        # This requires a recursive CTE (Common Table Expression) for SQLite 3.8.3+
        # If your SQLite is older, this will fail. Assuming modern SQLite:
        
        cur.execute("""
            WITH RECURSIVE SubCategories (id) AS (
                SELECT ? 
                UNION ALL 
                SELECT c.id FROM categories c JOIN SubCategories sc ON c.parent_id = sc.id
            )
            SELECT id FROM SubCategories
        """, (category_id,))
        
        descendant_ids = [row[0] for row in cur.fetchall()]
        
        if descendant_ids:
            placeholders = ",".join("?" * len(descendant_ids))
            sql += f"WHERE category_id IN ({placeholders}) "
            params.extend(descendant_ids)

    sql += "ORDER BY updated_at DESC LIMIT 50"
    
    cur.execute(sql, params)
    docs = cur.fetchall()
    
    categories = fetch_categories(conn)
    conn.close()
    
    return render_template("index.html", docs=docs, categories=categories, selected_category=category_id)

@app.route("/doc/<int:doc_id>")
def view_doc(doc_id):
    conn = get_db_conn() # Use the consistent get_db_conn() function
    cur = conn.cursor()
    
    # Fetch required metadata: id, title, filename, manual_relevance, category_id (for back button), access_count, file_path
    cur.execute("SELECT id, title, filename, manual_relevance, category_id, access_count, file_path FROM documents WHERE id=?", (doc_id,))
    doc = cur.fetchone()
    
    # 1. FIX: Removed the trailing backslash on the line below
    cur.execute("UPDATE documents SET access_count = access_count + 1 WHERE id=?", (doc_id,))
    
    # 2. This line no longer had the Syntax Error, but it should not have a backslash either
    cur.execute("INSERT INTO access_logs (document_id, event_type) VALUES (?, 'view')", (doc_id,))
    conn.commit()
    conn.close()
    
    # Pass only the document metadata (doc) to the template
    return render_template("doc.html", doc=doc)

@app.route("/rate", methods=["POST"])
def rate_doc():
    doc_id = int(request.form["doc_id"])
    stars = int(request.form["stars"])
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE documents SET manual_relevance = ? WHERE id = ?", (stars, doc_id))
    cur.execute("INSERT INTO access_logs (document_id, event_type) VALUES (?, 'rating')", (doc_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("view_doc", doc_id=doc_id))

@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q","")
    if not q:
        return redirect(url_for("index"))
    hits = retrieve(q, top_k=6)
    # optionally increment search logs
    return render_template("search.html", query=q, hits=hits)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
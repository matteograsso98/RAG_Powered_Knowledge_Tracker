# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import sqlite3
from retrieve_rag import retrieve, get_db_conn

app = Flask(__name__)

DB = "rag_library.db"

def db_conn():
    return sqlite3.connect(DB)

@app.route("/")
def index():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, filename, manual_relevance, auto_relevance, access_count FROM documents ORDER BY updated_at DESC LIMIT 50")
    docs = cur.fetchall()
    conn.close()
    return render_template("index.html", docs=docs)

@app.route("/doc/<int:doc_id>")
def view_doc(doc_id):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, filename, manual_relevance, auto_relevance, access_count FROM documents WHERE id=?", (doc_id,))
    doc = cur.fetchone()
    cur.execute("SELECT chunk_index, substr(text,1,800) FROM chunks WHERE document_id=? ORDER BY chunk_index LIMIT 10", (doc_id,))
    chunks = cur.fetchall()
    # increment access_count + log
    cur.execute("UPDATE documents SET access_count = access_count + 1 WHERE id=?", (doc_id,))
    cur.execute("INSERT INTO access_logs (document_id, event_type) VALUES (?, 'view')", (doc_id,))
    conn.commit()
    conn.close()
    return render_template("doc.html", doc=doc, chunks=chunks)

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

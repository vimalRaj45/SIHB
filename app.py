from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import requests
import bcrypt
from sentence_transformers import SentenceTransformer
import io
from flask import send_file
from pdf2image import convert_from_bytes
import pytesseract
from docx import Document
from io import BytesIO
import fitz  # pip install PyMuPDF
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# Load .env variables
load_dotenv()

# ==== DB CONFIG ====
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "sslmode": os.getenv("DB_SSLMODE", "require")
}

HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ==== Load local embedding model ====
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

# ==== DB HELPER ====
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# ==== USER AUTH ====
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data["username"]
    password = data["password"].encode("utf-8")
    role = data.get("role", "user")

    hashed = bcrypt.hashpw(password, bcrypt.gensalt()).decode("utf-8")

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, hashed, role)
        )
        conn.commit()
        return jsonify({"msg": "User registered"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        cur.close()
        conn.close()


# ==== DOWNLOAD / VIEW DOCUMENT ====
@app.route("/document/<int:doc_id>", methods=["GET"])
def get_document(doc_id):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT filename, file_data FROM documents WHERE id=%s", (doc_id,))
    doc = cur.fetchone()
    cur.close()
    conn.close()

    if not doc:
        return jsonify({"error": "Document not found"}), 404

    # Serve file to browser
    return send_file(
        io.BytesIO(doc["file_data"]),
        download_name=doc["filename"],
        as_attachment=False  # Set False if you want browser to try opening it
    )        

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data["username"]
    password = data["password"].encode("utf-8")

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if bcrypt.checkpw(password, user["password"].encode("utf-8")):
        return jsonify({"msg": "Login successful", "user_id": user["id"], "role": user["role"]})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# ==== Hugging Face Helpers ====
def hf_classify_document(text):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    candidate_labels = [
        "Incident Report", "Vendor Invoice", "Safety Bulletin",
        "HR Policy", "Engineering Drawing", "Regulatory Directive"
    ]
    payload = {"inputs": text[:1000], "parameters": {"candidate_labels": candidate_labels}}
    response = requests.post(url, headers=HF_HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Classification API Error: {response.status_code} - {response.text}")
    return response.json()["labels"][0]

def hf_summarize(text):
    url = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    payload = {"inputs": text[:1024]}
    response = requests.post(url, headers=HF_HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Summarization API Error: {response.status_code} - {response.text}")
    return response.json()[0]["summary_text"]

# ==== Local Embedding ====
def local_embed(text: str):
    if not text.strip():
        return [0.0] * 384
    return embed_model.encode([text])[0].tolist()  # 384-dim vector

@app.route("/upload", methods=["POST"])
def upload():
    user_id = request.form["user_id"]
    file = request.files["file"]
    filename = file.filename
    file_bytes = file.read()

    extracted_text = request.form.get("extracted_text", "").strip()

    # ---- Cloud-ready text extraction ----
    if not extracted_text:
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            # PyMuPDF: extract text from PDF pages
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            full_text = ""
            for page in doc:
                text = page.get_text()
                if text.strip():
                    full_text += text + "\n"
                else:
                    # If page is image-based, do OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    full_text += pytesseract.image_to_string(img) + "\n"
            extracted_text = full_text.strip()

        elif ext in ["docx"]:
            doc = Document(io.BytesIO(file_bytes))
            extracted_text = "\n".join([p.text for p in doc.paragraphs]).strip()

        elif ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            img = Image.open(io.BytesIO(file_bytes))
            extracted_text = pytesseract.image_to_string(img).strip()

        else:
            extracted_text = ""

    if not extracted_text:
        return jsonify({"error": "No text could be extracted"}), 400

    # ---- Hugging Face processing ----
    classification = hf_classify_document(extracted_text)
    summary = hf_summarize(extracted_text)
    embedding = local_embed(extracted_text)

    # ---- Save to DB ----
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents (user_id, filename, file_data, extracted_text, summary, classification, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (user_id, filename, psycopg2.Binary(file_bytes), extracted_text, summary, classification, embedding))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"msg": "File uploaded & processed", "extracted_text": extracted_text}), 201
# ==== SEARCH ====
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    user_id = data["user_id"]
    role = data["role"]
    query = data["query"]

    query_emb = local_embed(query)  # Local embedding

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if role == "admin":
        cur.execute("""
            SELECT id, filename, summary, classification, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY similarity DESC LIMIT 5
        """, (query_emb,))
    else:
        cur.execute("""
            SELECT id, filename, summary, classification, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE user_id = %s
            ORDER BY similarity DESC LIMIT 5
        """, (query_emb, user_id))

    results = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([dict(r) for r in results])

# ==== RUN APP ====
if __name__ == "__main__":
    app.run(debug=True)

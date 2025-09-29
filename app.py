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
from PIL import Image
from flask import Flask, jsonify
import imaplib
import email
from email.header import decode_header

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
        return jsonify({"msg": "Login successful", "user_id": user["id"], "role": user["role"],"username": user["username"]})
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



# ==== CONFIG ====
EMAIL = "vimalraj5207@gmail.com"
PASSWORD = "lslenevpvcxrvzjt"  # Gmail app password
IMAP_SERVER = "imap.gmail.com"

def fetch_last_5_emails():
    # Connect to Gmail IMAP server
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    # Search all emails
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()

    # Get last 5 email IDs (latest ones)
    email_ids = email_ids[-5:]

    emails = []

    for e_id in reversed(email_ids):  # reverse so newest comes first
        status, msg_data = mail.fetch(e_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Decode subject safely
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    try:
                        subject = subject.decode(encoding or "utf-8")
                    except:
                        subject = subject.decode("utf-8", errors="ignore")

                from_ = msg.get("From")

                # Get email body safely
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

                emails.append({
                    "from": from_,
                    "subject": subject,
                    "body": body[:500]  # Limit body to 500 chars
                })

    mail.logout()
    return emails


# ==== ENDPOINT ====
@app.route("/last_emails", methods=["GET"])
def get_last_emails():
    try:
        emails = fetch_last_5_emails()
        return jsonify({"count": len(emails), "emails": emails})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




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


@app.route("/profile/<int:user_id>", methods=["GET"])
def profile(user_id):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # Fetch user info
        cur.execute("SELECT id, username, role FROM users WHERE id=%s", (user_id,))
        user = cur.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Count documents uploaded by user
        cur.execute("SELECT COUNT(*) AS doc_count FROM documents WHERE user_id=%s", (user_id,))
        doc_count = cur.fetchone()["doc_count"]

        # Optional: total documents
        cur.execute("SELECT COUNT(*) AS total_docs FROM documents")
        total_docs = cur.fetchone()["total_docs"]

        # Fetch user's documents with filename, summary, classification
        cur.execute("""
            SELECT id, filename, summary, classification
            FROM documents
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        documents = cur.fetchall()
        documents_list = [dict(doc) for doc in documents]

        profile_data = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "uploaded_docs": doc_count,
            "total_docs": total_docs,
            "documents": documents_list
        }

        return jsonify(profile_data)

    finally:
        cur.close()
        conn.close()



@app.route("/recent_docs", methods=["GET"])
def recent_docs():
    """
    Returns recent document uploads with user info, summary, and timestamp.
    """
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute("""
            SELECT d.id, d.filename, d.extracted_text, d.summary, d.classification,
                   d.created_at, u.username
            FROM documents d
            JOIN users u ON d.user_id = u.id
            ORDER BY d.created_at DESC
            LIMIT 20;
        """)

        docs = cur.fetchall()
        cur.close()
        conn.close()

        # Convert to JSON-serializable format
        docs_list = [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "summary": doc["summary"],
                "classification": doc["classification"],
                "created_at": doc["created_at"].isoformat(),
                "uploaded_by": doc["username"]
            }
            for doc in docs
        ]

        return jsonify({"documents": docs_list}), 200

    except Exception as e:
        print("Error fetching recent documents:", e)
        return jsonify({"error": "Failed to fetch recent documents"}), 500

@app.route("/notifications", methods=["GET"])
def notifications():
    search = request.args.get("search", "").strip()
    user_filter = request.args.get("user", "").strip()
    class_filter = request.args.get("classification", "").strip()
    date_filter = request.args.get("date", "").strip()

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    base_query = """
        SELECT d.id, d.filename, d.summary, d.classification, d.created_at,
               u.username AS uploaded_by
        FROM documents d
        JOIN users u ON d.user_id = u.id
        WHERE 1=1
    """
    params = []

    if search:
        base_query += " AND (d.filename ILIKE %s OR d.summary ILIKE %s)"
        search_term = f"%{search}%"
        params.extend([search_term, search_term])

    if user_filter:
        base_query += " AND u.username ILIKE %s"
        params.append(f"%{user_filter}%")

    if class_filter:
        base_query += " AND d.classification = %s"
        params.append(class_filter)

    if date_filter:
        base_query += " AND DATE(d.created_at) = %s"
        params.append(date_filter)

    base_query += " ORDER BY d.created_at DESC LIMIT 50"

    cur.execute(base_query, tuple(params))
    results = cur.fetchall()
    cur.close()
    conn.close()

    notifications = []
    for row in results:
        notifications.append({
            "doc_id": row["id"],
            "filename": row["filename"],
            "summary": row["summary"],
            "classification": row["classification"],
            "uploaded_by": row["uploaded_by"],
            "created_at": row["created_at"],
            "file_url": f"/document/{row['id']}"
        })

    return jsonify(notifications)



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


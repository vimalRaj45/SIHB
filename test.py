from flask import Flask, jsonify
import imaplib
import email
from email.header import decode_header

app = Flask(__name__)

# ==== CONFIG ====
EMAIL = "vimalraj5207@gmail.com"
PASSWORD = "lslenevpvcxrvzjt"  # Your Gmail app password
IMAP_SERVER = "imap.gmail.com"

def fetch_unread_emails():
    # Connect to Gmail IMAP server
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    # Search for UNSEEN (unread) emails
    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    # Only take first 5 unread emails
    email_ids = email_ids[:5]

    unread_emails = []

    for e_id in email_ids:
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

                unread_emails.append({
                    "from": from_,
                    "subject": subject,
                    "body": body[:500]  # Limit body to 500 chars
                })

    mail.logout()
    return unread_emails

# ==== ENDPOINT ====
@app.route("/unread_emails", methods=["GET"])
def get_unread_emails():
    try:
        emails = fetch_unread_emails()
        return jsonify({"count": len(emails), "emails": emails})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import streamlit as st
import pandas as pd
import re
import joblib
from io import StringIO
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Log Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .hero-card {
        background: linear-gradient(135deg, #dbeafe 0%, #ccfbf1 100%);
        border: 1px solid rgba(255,255,255,0.7);
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.75rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #334155;
        font-size: 1.1rem;
        line-height: 1.8;
        max-width: 900px;
    }

    .section-card {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 22px;
        padding: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        backdrop-filter: blur(10px);
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 20px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }

    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }

    .badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.35rem;
        margin-right: 0.35rem;
    }

    .badge-blue {
        background: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    .badge-green {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .badge-purple {
        background: #ede9fe;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }

    .small-note {
        color: #475569;
        font-size: 0.95rem;
    }

    div[data-testid="stFileUploader"] {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 16px;
        border: 1.5px dashed #cbd5e1;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.18);
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.75rem 1rem;
        box-shadow: 0 10px 24px rgba(37,99,235,0.22);
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #0284c7);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")         # e.g. your Gmail
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")     # e.g. Gmail app password
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))

# -----------------------------
# Cached Models
# -----------------------------
@st.cache_resource
def load_bert_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = joblib.load("models/log_classifier.joblib")
    return embedder, classifier

@st.cache_resource
def load_groq_client():
    return Groq()

# -----------------------------
# Classification Functions
# -----------------------------
def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, str(log_message)):
            return label
    return "Unclassified"

def classify_with_bert(log_message):
    try:
        model_embedding, model_classification = load_bert_resources()
        embeddings = model_embedding.encode([str(log_message)])
        probabilities = model_classification.predict_proba(embeddings)[0]
        if max(probabilities) < 0.5:
            return "Unclassified"
        predicted_label = model_classification.predict(embeddings)[0]
        return predicted_label
    except Exception as e:
        return f"Error: {e}"

def classify_with_llm(log_msg):
    try:
        groq_client = load_groq_client()
        prompt = f"""
Classify the log message into one of these categories:
(1) Workflow Error
(2) Deprecation Warning

If you cannot determine a category, return:
Unclassified

Return only the category name and nothing else.

Log message: {log_msg}
"""
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        content = chat_completion.choices[0].message.content.strip()
        allowed_labels = ["Workflow Error", "Deprecation Warning", "Unclassified"]
        if content in allowed_labels:
            return content
        for label in allowed_labels:
            if label.lower() in content.lower():
                return label
        return "Unclassified"
    except Exception as e:
        return f"Error: {e}"

def run_classifier(log_text, method):
    if method == "Regex":
        return classify_with_regex(log_text)
    elif method == "BERT":
        return classify_with_bert(log_text)
    elif method == "LLM":
        return classify_with_llm(log_text)
    return "Unclassified"

def get_status(label):
    if str(label).startswith("Error:"):
        return "System Error"
    return "Matched" if label != "Unclassified" else "Needs Review"

# -----------------------------
# Email helper
# -----------------------------
def send_results_email(to_email: str, subject: str, body: str, csv_bytes: bytes, filename: str = "classified_output.csv"):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={filename}")
    msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

# -----------------------------
# UI Header
# -----------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧠 Log Classification System</div>
    <div class="hero-subtitle">
        Upload log files, classify entries using Regex, BERT, or LLM models,
        review predictions in a clean interactive table, and export the final
        output CSV for analysis or reporting.
    </div>
    <div style="margin-top: 1rem;">
        <span class="badge badge-blue">NLP Project</span>
        <span class="badge badge-green">Streamlit UI</span>
        <span class="badge badge-purple">BERT + LLM + Regex</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    classifier_choice = st.selectbox(
        "Choose classification method",
        ["Regex", "BERT", "LLM"]
    )
    st.markdown("### Active Selection")
    st.info(f"Model: {classifier_choice}")
    st.markdown("### Supported File Types")
    st.write("- CSV with `log_message` column")
    st.write("- TXT file with one log per line")
    st.markdown("### Notes")
    st.write("- Regex is fastest.")
    st.write("- BERT uses your trained model.")
    st.write("- LLM needs Groq API access from `.env`.")
    st.markdown("---")
    st.markdown("### Email Settings")
    st.caption("Configure EMAIL_SENDER, EMAIL_PASSWORD etc. in your .env.")

# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Upload Test File")
st.caption("Upload a CSV file with a `log_message` column or a TXT file containing one log message per line.")
uploaded_file = st.file_uploader(
    "Choose a CSV or TXT file",
    type=["csv", "txt"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Processing
# -----------------------------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "log_message" not in df.columns:
                st.error("CSV must contain a column named 'log_message'")
                st.stop()
        else:
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            logs = [line.strip() for line in string_data.splitlines() if line.strip()]
            df = pd.DataFrame({"log_message": logs})

        with st.spinner("Classifying logs..."):
            df["log_message"] = df["log_message"].astype(str)
            df["predicted_label"] = df["log_message"].apply(lambda x: run_classifier(x, classifier_choice))
            df["classifier_used"] = classifier_choice
            df["status"] = df["predicted_label"].apply(get_status)

        total_logs = len(df)
        classified_logs = (df["status"] == "Matched").sum()
        review_logs = (df["status"] == "Needs Review").sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Logs</div>
                <div class="metric-value">{total_logs}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classified</div>
                <div class="metric-value">{classified_logs}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Needs Review</div>
                <div class="metric-value">{review_logs}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Classification Results")
        st.caption("Preview the structured output before downloading the final result file.")
        st.dataframe(df, use_container_width=True, height=420)
        st.markdown("</div>", unsafe_allow_html=True)

        csv_output = df.to_csv(index=False).encode("utf-8")

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Download & Email Output")
        st.markdown(
            "<p class='small-note'>Download the CSV or send it directly via email as an attachment.</p>",
            unsafe_allow_html=True
        )

        col_dl, col_email = st.columns(2)

        with col_dl:
            st.download_button(
                label="📥 Download Output CSV",
                data=csv_output,
                file_name="classified_output.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_email:
            recv_email = st.text_input("Recipient email", placeholder="user@example.com")
            email_subject = st.text_input("Email subject", value="Log Classification Results")
            email_body = st.text_area(
                "Email message",
                value="Hi,\n\nPlease find attached the classified log output.\n\nRegards,"
            )
            send_btn = st.button("📧 Send Email with Attachment", use_container_width=True)

            if send_btn:
                if not EMAIL_SENDER or not EMAIL_PASSWORD:
                    st.error("EMAIL_SENDER or EMAIL_PASSWORD not set in .env")
                elif not recv_email:
                    st.error("Please enter a recipient email address.")
                else:
                    try:
                        send_results_email(
                            to_email=recv_email,
                            subject=email_subject,
                            body=email_body,
                            csv_bytes=csv_output
                        )
                        st.success(f"Email sent successfully to {recv_email}!")
                    except Exception as e:
                        st.error(f"Failed to send email: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV or TXT file to begin log classification.")
import re
import json
import smtplib
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import faiss
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "zsf_logs"
INPUT_FILE = LOG_DIR / "engine.log"

START_TIME_STR = "2026-02-12 03:00:00"
END_TIME_STR = "2026-02-12 04:00:00"

START_TIME = datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M:%S")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Hardcoded email settings
SENDER_EMAIL = "your_email_address"
SENDER_PASSWORD = "your_app_password"   # Gmail App Password, not normal password
RECIPIENT_EMAIL = "sender_email_address"  # For testing, send to self

# Hardcoded OpenAI API key
OPENAI_API_KEY = "your_openai_api_key_here"  # Replace with your actual OpenAI API key

# Models
OPENAI_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG knowledge base + FAISS files
RAG_KB_FILE = BASE_DIR / "rag_knowledge_base.json"
FAISS_INDEX_FILE = BASE_DIR / "rag_faiss.index"
FAISS_METADATA_FILE = BASE_DIR / "rag_faiss_metadata.json"
RAG_TOP_K = 3


# =========================================================
# REGEX PATTERNS
# =========================================================

AUDIT_PATTERN = re.compile(
    r"""
    ^
    (?P<time_stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})
    ,\d+(?:[-+]\d{2})?
    \s+
    (?P<event>[A-Z]+)
    .*?
    EVENT_ID:\s+
    (?P<event_id>[A-Z0-9_]+\([^)]*\))
    ,
    .*?
    \bon\s+host\s+(?P<host_name>[A-Za-z0-9._-]+)
    .*$
    """,
    re.VERBOSE,
)

ERROR_PATTERN = re.compile(
    r"""
    ^
    (?P<time_stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})
    ,\d+(?:[-+]\d{2})?
    \s+
    (?P<event>ERROR)
    \s+
    \[(?P<source>[^\]]+)\]
    .*?
    failed:\s*(?P<error_message>.*)$
    """,
    re.VERBOSE,
)


# =========================================================
# PARSING
# =========================================================

def parse_line(line: str) -> dict | None:
    """
    Parse one log line.
    Supports:
      1) EVENT_ID audit lines
      2) Generic ERROR lines
    """
    audit_match = AUDIT_PATTERN.search(line)
    if audit_match:
        return {
            "time_stamp": audit_match.group("time_stamp"),
            "event": audit_match.group("event"),
            "event_id": audit_match.group("event_id"),
            "host_name": audit_match.group("host_name"),
            "source": "N/A",
            "error_message": "",
            "log_type": "audit",
            "log": line,
        }

    error_match = ERROR_PATTERN.search(line)
    if error_match:
        return {
            "time_stamp": error_match.group("time_stamp"),
            "event": error_match.group("event"),
            "event_id": "N/A",
            "host_name": "N/A",
            "source": error_match.group("source"),
            "error_message": error_match.group("error_message"),
            "log_type": "error",
            "log": line,
        }

    return None


def parse_and_filter(input_file: Path) -> list[dict]:
    """
    Read the log file and return matched records within the time window.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Log file not found: {input_file}")

    results: list[dict] = []

    with input_file.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            parsed = parse_line(line)
            if not parsed:
                continue

            ts = datetime.strptime(parsed["time_stamp"], "%Y-%m-%d %H:%M:%S")
            if not (START_TIME <= ts <= END_TIME):
                continue

            results.append(parsed)

    return results


def get_error_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["event"] == "ERROR"]


# =========================================================
# RAG: KNOWLEDGE BASE + EMBEDDINGS + RETRIEVAL
# =========================================================

def create_default_knowledge_base() -> None:
    """
    Creates a small starter knowledge base if rag_knowledge_base.json does not exist yet.
    You can add more real incidents later.
    """
    if RAG_KB_FILE.exists():
        return

    examples = [
        {
            "id": "kb-001",
            "title": "Host communication failure",
            "problem": "Engine cannot communicate with host or host operation failed.",
            "root_cause": "Host agent may be down, host may be unreachable, DNS may be wrong, or SSH/service connectivity may be blocked.",
            "checks": [
                "Ping the host from the engine server.",
                "Check DNS resolution for the host FQDN.",
                "Check host agent/service status.",
                "Check firewall rules between engine and host."
            ],
            "fix": "Restart the host agent/service, fix DNS/network connectivity, then retry the operation from the engine.",
            "prevention": "Monitor host heartbeat and alert when host management service stops."
        },
        {
            "id": "kb-002",
            "title": "Storage or multipath issue",
            "problem": "Log mentions faulty multipath, storage path failure, or storage domain access issue.",
            "root_cause": "One or more SAN/iSCSI/FC paths may be down, storage target may be unreachable, or multipath configuration may be incorrect.",
            "checks": [
                "Check multipath -ll on the host.",
                "Check storage network connectivity.",
                "Check storage array/target availability.",
                "Review recent storage zoning or network changes."
            ],
            "fix": "Restore failed storage paths, restart multipath if needed, and verify the storage domain is active.",
            "prevention": "Monitor path count and alert when redundant paths drop below expected count."
        },
        {
            "id": "kb-003",
            "title": "VM provisioning failure",
            "problem": "VM creation, reimage, or provisioning workflow failed.",
            "root_cause": "Possible missing template, invalid VM parameters, unavailable host capacity, network profile mismatch, or backend automation error.",
            "checks": [
                "Validate VM shape, location, and network profile.",
                "Check template/image availability.",
                "Check host cluster capacity.",
                "Review backend automation logs."
            ],
            "fix": "Correct the input parameters or template mapping, then rerun the provisioning workflow.",
            "prevention": "Add pre-flight validation before provisioning starts."
        }
    ]

    with RAG_KB_FILE.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def kb_item_to_text(item: dict) -> str:
    """
    Converts one knowledge-base item into searchable text.
    """
    checks = item.get("checks", [])
    if isinstance(checks, list):
        checks_text = " ".join(checks)
    else:
        checks_text = str(checks)

    return (
        f"Title: {item.get('title', '')}\n"
        f"Problem: {item.get('problem', '')}\n"
        f"Root cause: {item.get('root_cause', '')}\n"
        f"Checks: {checks_text}\n"
        f"Fix: {item.get('fix', '')}\n"
        f"Prevention: {item.get('prevention', '')}"
    )


def load_knowledge_base() -> list[dict]:
    create_default_knowledge_base()
    with RAG_KB_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """
    Create an embedding vector for text using OpenAI embeddings.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors so FAISS inner-product search behaves like cosine similarity.
    """
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_faiss_index(client: OpenAI) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Builds a FAISS vector index from rag_knowledge_base.json.

    Files created:
      - rag_faiss.index          FAISS vector index
      - rag_faiss_metadata.json  Metadata for each vector

    Delete both files if you update rag_knowledge_base.json and want to rebuild.
    """
    kb_items = load_knowledge_base()

    metadata = []
    embeddings = []

    for item in kb_items:
        text = kb_item_to_text(item)
        embedding = get_embedding(client, text)
        embeddings.append(embedding)
        metadata.append(
            {
                "id": item.get("id"),
                "text": text,
                "item": item,
            }
        )

    embedding_matrix = np.array(embeddings, dtype="float32")
    embedding_matrix = normalize_vectors(embedding_matrix)

    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with FAISS_METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return index, metadata


def load_or_build_faiss_index(client: OpenAI) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Loads FAISS index from disk if available. Otherwise builds it.
    """
    if FAISS_INDEX_FILE.exists() and FAISS_METADATA_FILE.exists():
        index = faiss.read_index(str(FAISS_INDEX_FILE))
        with FAISS_METADATA_FILE.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    return build_faiss_index(client)


def build_error_query(error_records: list[dict]) -> str:
    """
    Converts current ERROR logs into one query for RAG similarity search.
    """
    return "\n".join(
        [
            f"Time: {r['time_stamp']}\n"
            f"Source: {r['source']}\n"
            f"Message: {r['error_message']}\n"
            f"Full Log: {r['log']}"
            for r in error_records[:10]
        ]
    )


def retrieve_similar_incidents(client: OpenAI, error_records: list[dict], top_k: int = RAG_TOP_K) -> list[dict]:
    """
    RAG retrieval step using FAISS:
    1. Embed the current error logs
    2. Search FAISS for similar historical incidents
    3. Return the top matching incidents
    """
    if not error_records:
        return []

    index, metadata = load_or_build_faiss_index(client)

    query_text = build_error_query(error_records)
    query_embedding = np.array([get_embedding(client, query_text)], dtype="float32")
    query_embedding = normalize_vectors(query_embedding)

    search_k = min(top_k, len(metadata))
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = metadata[idx]
        results.append(
            {
                "score": float(score),
                "item": row["item"],
                "text": row["text"],
            }
        )

    return results


def format_rag_context(similar_incidents: list[dict]) -> str:
    """
    Formats retrieved incidents for the LLM prompt.
    """
    if not similar_incidents:
        return "No similar incidents were retrieved."

    blocks = []
    for incident in similar_incidents:
        item = incident["item"]
        checks = item.get("checks", [])
        if isinstance(checks, list):
            checks_html = "".join([f"<li>{c}</li>" for c in checks])
        else:
            checks_html = f"<li>{checks}</li>"

        blocks.append(
            f"""
            <p><b>Similarity score:</b> {incident['score']:.3f}</p>
            <p><b>Past incident:</b> {item.get('title', '')}</p>
            <p><b>Known root cause:</b> {item.get('root_cause', '')}</p>
            <p><b>Known checks:</b></p>
            <ul>{checks_html}</ul>
            <p><b>Known fix:</b> {item.get('fix', '')}</p>
            <p><b>Prevention:</b> {item.get('prevention', '')}</p>
            """
        )

    return "\n".join(blocks)


# =========================================================
# OPENAI ANALYSIS WITH RAG
# =========================================================

def ask_openai_for_solution(error_records: list[dict]) -> tuple[str, str]:
    """
    Send ERROR logs + retrieved similar incidents to OpenAI and get troubleshooting guidance back as HTML.
    Returns:
      1. AI analysis HTML
      2. RAG context HTML
    """
    if not error_records:
        return "<p>No ERROR logs found. AI troubleshooting was skipped.</p>", "<p>No RAG lookup needed.</p>"

    client = OpenAI(api_key=OPENAI_API_KEY)

    log_block = build_error_query(error_records)

    try:
        similar_incidents = retrieve_similar_incidents(client, error_records)
        rag_context_html = format_rag_context(similar_incidents)

        prompt = f"""
You are an infrastructure troubleshooting assistant.

Analyze the current ERROR logs. Use the retrieved historical incidents as reference context.
Do not blindly copy the historical fix. Compare it against the current log evidence.

Return the answer in simple HTML using only:
<p>, <ul>, <li>, <b>

Current ERROR logs:
{log_block}

Retrieved historical incidents:
{rag_context_html}

Provide:
1. Likely root cause
2. Why the retrieved incidents are or are not relevant
3. Immediate checks
4. Suggested fix
5. Prevention ideas
""".strip()

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions="You are an expert in infrastructure operations, automation, and log troubleshooting.",
            input=prompt,
        )

        return response.output_text or "<p>No analysis returned.</p>", rag_context_html

    except Exception as e:
        return f"<p><b>OpenAI/RAG analysis failed:</b> {str(e)}</p>", "<p>RAG lookup failed.</p>"


# =========================================================
# EMAIL FORMATTING
# =========================================================

def build_table_html(records: list[dict]) -> str:
    if not records:
        return "<p>No matching logs found in the selected time range.</p>"

    rows = []
    for r in records:
        rows.append(
            f"""
            <tr>
                <td>{r['time_stamp']}</td>
                <td>{r['event']}</td>
                <td>{r['event_id']}</td>
                <td>{r['host_name']}</td>
                <td>{r['source']}</td>
                <td>{r['error_message']}</td>
            </tr>
            """
        )

    return f"""
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
        <tr>
            <th>Time</th>
            <th>Event</th>
            <th>Event ID</th>
            <th>Host</th>
            <th>Source</th>
            <th>Error Message</th>
        </tr>
        {''.join(rows)}
    </table>
    """


def build_email_message(records: list[dict], ai_analysis_html: str, rag_context_html: str):
    total_count = len(records)
    error_count = len(get_error_records(records))
    logs_table = build_table_html(records)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Log Report</title>
    </head>
    <body style="margin:0; padding:0; background-color:#f4f4f4; font-family:Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f4f4f4; padding:20px 0;">
            <tr>
                <td align="center">
                    <table width="900" cellpadding="0" cellspacing="0" border="0" style="background:#ffffff; border-radius:8px; overflow:hidden;">
                        <tr>
                            <td style="background:#2d6cdf; color:#ffffff; padding:20px; text-align:center; font-size:24px; font-weight:bold;">
                                Log Report with RAG AI Analysis
                            </td>
                        </tr>

                        <tr>
                            <td style="padding:30px; color:#333333; font-size:16px; line-height:1.6;">
                                <p><b>Time range:</b> {START_TIME_STR} → {END_TIME_STR}</p>
                                <p><b>Total matched logs:</b> {total_count}</p>
                                <p><b>ERROR count:</b> {error_count}</p>

                                <h3>Matched Logs</h3>
                                {logs_table}

                                <h3>Retrieved Similar Incidents</h3>
                                {rag_context_html}

                                <h3>AI Troubleshooting with RAG</h3>
                                {ai_analysis_html}

                                <p>Best regards,<br>Automated Log Monitor</p>
                            </td>
                        </tr>

                        <tr>
                            <td style="background:#f0f0f0; padding:15px; text-align:center; font-size:12px; color:#777777;">
                                This is an automated email. Please do not reply directly.
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["From"] = SENDER_EMAIL
    message["To"] = RECIPIENT_EMAIL
    message["Subject"] = "Log Report with RAG AI Analysis"
    message.attach(MIMEText(html, "html"))

    return message


# =========================================================
# SEND EMAIL
# =========================================================

def send_email(message):
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())


# =========================================================
# MAIN
# =========================================================

def main():
    records = parse_and_filter(INPUT_FILE)
    error_records = get_error_records(records)

    ai_analysis_html, rag_context_html = ask_openai_for_solution(error_records)
    message = build_email_message(records, ai_analysis_html, rag_context_html)

    send_email(message)

    print(f"Matched records: {len(records)}")
    print(f"ERROR records: {len(error_records)}")
    print("Email sent successfully.")
    print(f"RAG knowledge base: {RAG_KB_FILE}")
    print(f"FAISS index: {FAISS_INDEX_FILE}")
    print(f"FAISS metadata: {FAISS_METADATA_FILE}")


if __name__ == "__main__":
    main()

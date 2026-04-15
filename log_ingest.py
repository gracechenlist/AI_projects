import re
import smtplib
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

# Model name
OPENAI_MODEL = "gpt-5.2"

# =========================================================
# REGEX PATTERNS
# =========================================================

# Original EVENT_ID-style audit line
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

# Generic ERROR line
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
# OPENAI ANALYSIS
# =========================================================

def ask_openai_for_solution(error_records: list[dict]) -> str:
    """
    Send ERROR logs to OpenAI and get troubleshooting guidance back as HTML.
    """
    if not error_records:
        return "<p>No ERROR logs found. AI troubleshooting was skipped.</p>"

    client = OpenAI(api_key=OPENAI_API_KEY)

    log_block = "\n".join(
        [
            f"- Time: {r['time_stamp']}\n"
            f"  Source: {r['source']}\n"
            f"  Message: {r['error_message']}\n"
            f"  Full Log: {r['log']}"
            for r in error_records[:10]
        ]
    )

    prompt = f"""
You are an infrastructure troubleshooting assistant.

Analyze these ERROR logs and provide:
1. Likely root cause
2. Immediate checks
3. Suggested fix
4. Prevention ideas

Return the answer in simple HTML using only:
<p>, <ul>, <li>, <b>

Logs:
{log_block}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions="You are an expert in infrastructure operations, automation, and log troubleshooting.",
            input=prompt,
        )
        return response.output_text or "<p>No analysis returned.</p>"
    except Exception as e:
        return f"<p><b>OpenAI analysis failed:</b> {str(e)}</p>"


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


def build_email_message(records: list[dict], ai_analysis_html: str):
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
                                Log Report with AI Analysis
                            </td>
                        </tr>

                        <tr>
                            <td style="padding:30px; color:#333333; font-size:16px; line-height:1.6;">
                                <p><b>Time range:</b> {START_TIME_STR} → {END_TIME_STR}</p>
                                <p><b>Total matched logs:</b> {total_count}</p>
                                <p><b>ERROR count:</b> {error_count}</p>

                                <h3>Matched Logs</h3>
                                {logs_table}

                                <h3>AI Troubleshooting</h3>
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
    message["Subject"] = "Log Report with AI Analysis"
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

    ai_analysis_html = ask_openai_for_solution(error_records)
    message = build_email_message(records, ai_analysis_html)

    send_email(message)

    print(f"Matched records: {len(records)}")
    print(f"ERROR records: {len(error_records)}")
    print("Email sent successfully.")


if __name__ == "__main__":
    main()
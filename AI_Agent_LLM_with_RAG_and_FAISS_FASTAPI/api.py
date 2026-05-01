from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File

from log_ingest_rag_faiss import (
    parse_and_filter,
    get_error_records,
    ask_openai_for_solution,
    search_similar_logs,
    load_or_build_faiss_index,
    OpenAI,
    OPENAI_API_KEY,
)

app = FastAPI(title="AI Log Analysis API")

index = None
metadata = None


@app.on_event("startup")
def startup_event():
    global index, metadata
    client = OpenAI(api_key=OPENAI_API_KEY)
    index, metadata = load_or_build_faiss_index(client)


@app.get("/")
def root():
    return {"message": "AI Log Analysis API is running"}


@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        records = parse_and_filter(temp_path)
        error_records = get_error_records(records)

        similar_incidents = []
        for err in error_records[:3]:
            query_text = err.get("error_message") or err.get("log") or ""
            similar_incidents.append(
                search_similar_logs(query_text, index=index, metadata=metadata, k=3)
            )

        ai_analysis_html, rag_context_html = ask_openai_for_solution(error_records)

        return {
            "total_logs": len(records),
            "error_count": len(error_records),
            "errors": error_records,
            "similar_incidents": similar_incidents,
            "rag_context_html": rag_context_html,
            "ai_analysis_html": ai_analysis_html,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

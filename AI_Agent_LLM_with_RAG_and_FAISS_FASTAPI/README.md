# AI-Powered Log Analysis with RAG and FAISS

This project is an AI-powered log monitoring tool that parses infrastructure logs, detects errors, retrieves similar historical incidents using FAISS vector search, and uses an OpenAI LLM to generate troubleshooting recommendations.

## Features

- Parses log files from `zsf_logs/engine.log`
- Extracts timestamp, event type, event ID, host name, source, and error message
- Filters logs by a defined time range
- Detects ERROR logs
- Converts logs into embeddings using OpenAI
- Stores and searches embeddings using FAISS
- Retrieves similar historical incidents with RAG
- Generates AI troubleshooting guidance
- Sends an HTML email report

## Tech Stack

- Python
- OpenAI API
- FAISS
- NumPy
- Regex
- SMTP / Gmail
- HTML email formatting

## Architecture

```text
Log File
   ↓
Regex Parser
   ↓
Time Filter
   ↓
ERROR Detection
   ↓
OpenAI Embeddings
   ↓
FAISS Vector Search
   ↓
Retrieve Similar Incidents
   ↓
LLM Troubleshooting Analysis
   ↓
HTML Email Report
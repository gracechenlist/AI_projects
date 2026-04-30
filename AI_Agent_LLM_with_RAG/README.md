# AI-Powered Log Monitoring with RAG

An intelligent log monitoring and alerting system that combines log parsing, automation, and Generative AI (LLM + RAG) to analyze infrastructure errors and provide actionable troubleshooting insights.

# Features
- Parse structured and unstructured logs (EVENT_ID + ERROR)
- Filter logs within a custom time window
- Use LLM to analyze error logs and suggest:
    Root cause
    Immediate checks
    Fix recommendations
    Prevention strategies
- RAG (Retrieval-Augmented Generation):
    Store historical incidents
    Retrieve similar past errors
    Improve AI accuracy with context
- Send formatted HTML email reports automatically
# AI_projects
Log Monitoring & AI Troubleshooting System

This project is a Python-based automation tool that:

Parses system logs (oVirt / datacenter logs)
Filters events within a specific time window
Extracts structured data (timestamp, event, event_id, host, etc.)
Detects ERROR logs
Uses the OpenAI API to generate troubleshooting insights
Sends a formatted HTML report via email

Features
-- Regex-based log parsing (supports multiple log formats)
-- Time-range filtering
-- Automatic detection of ERROR events
-- AI-powered root cause analysis and remediation suggestions
-- HTML email reporting (table + AI insights)
-- Modular and extensible design
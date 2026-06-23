# Hotel WhatsApp Agent

Backend service in Python for a hotel WhatsApp assistant using:

- `FastAPI` for the webhook
- `LangGraph` for the agent flow
- `OpenRouter` for text, image, and audio reasoning
- `EvolutionAPI` for WhatsApp send/receive
- `SQLite` for chat history, booking requests, and handoff tracking

## Current capabilities

- Handles text messages from EvolutionAPI
- Handles voice messages and transcribes them before they enter the agent
- Handles images and extracts useful context before the agent responds
- Detects and responds in Arabic, English, Urdu, Indonesian, and Turkish
- Answers FAQ from a local hotel knowledge file
- Answers trial pricing questions from the local knowledge base
- Collects booking details over chat
- Saves confirmed booking requests in SQLite and CSV
- Creates staff handoff records when the user asks for a human or the answer is uncertain
- Tracks analytics events and repeated missing-knowledge questions

## Project structure

```text
app/
  agent/
    graph.py
    prompts.py
  knowledge/
    default_hotel.py
  services/
    evolution.py
    knowledge.py
    media.py
    openrouter.py
    payload_parser.py
    storage.py
  config.py
  logging_utils.py
  main.py
  models.py
data/
  knowledge/
  exports/
logs/
```

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

## Webhook setup

Point EvolutionAPI to:

```text
POST /webhook/whatsapp
```

Example production URL:

```text
https://your-domain.com/webhook/whatsapp
```

## Deployment

This project is ready for Docker-based deployment on Coolify.

## Langfuse

Tracing is already wired into the code, but it only becomes active when these env vars are set:

```text
LANGFUSE_HOST=https://your-langfuse-domain.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

You can also use `LANGFUSE_BASE_URL` if that is the variable name you already use in Coolify.

Once those values are present, the app sends traces for:

- the incoming WhatsApp webhook
- audio transcription calls
- image analysis calls
- the main booking/FAQ decision call

This makes it easier to debug:

- why the agent gave a reply
- which prompt/context it saw
- where an error happened
- how a booking request moved through the flow

## Admin analytics

Set `ADMIN_API_KEY` in `.env`, then open:

```text
GET /admin?admin_key=your-admin-key
```

Use the same value in the `x-admin-key` header for API endpoints:

```text
GET /admin/api/analytics/summary
GET /admin/api/questions
GET /admin/api/messages
GET /admin/api/knowledge-gaps
```

The dashboard shows monthly conversation volume, booking leads, handoffs,
average response time, common questions, detected languages, and knowledge gaps.

## Notes

- Update `data/knowledge/hafawah.md` with the real hotel information.
- Trial prices are included for testing only. Replace them with real hotel rates before production.
- Replace all example secrets in `.env`.

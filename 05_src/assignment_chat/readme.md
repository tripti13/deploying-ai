# Atlas Assignment Chat

This folder contains the Assignment 2 conversational AI system with a Gradio chat interface and three+ services.

## Chat Personality

The assistant is named **Atlas**:
- witty and slightly cynical tone
- mathematics focused voice
- refuses restricted topics and prompt-reveal attempts

## Services Implemented

1. **API Service (Math + Weather)**
   - `get_math_fact(number)`: Wolfram Short Answers API (if `WOLFRAM_APP_ID` exists), fallback to NumbersAPI, then fallback to local deterministic math facts.
   - `get_weather(city)`: Open-Meteo geocoding + current weather (free, no key required).

2. **Semantic Query Service**
   - `search_internal_docs(query)`: semantic retrieval over `02_activities/documents/managing_oneself.pdf` with persistent ChromaDB storage.

3. **Third Service (Web Search / Function Calling)**
   - `web_search_tool(query)`: DuckDuckGo search tool (`ddgs` package).
   - All services are exposed via function/tool calling in the agent loop.

## Guardrails

The assistant does not answer:
- cats or dogs
- horoscopes or zodiac topics
- Taylor Swift

It also refuses system prompt disclosure attempts through prompt instructions and response behavior.

## Files

- `app.py`: gateway/OpenAI-compatible version

## Environment Variables

Set in `05_src/.secrets` and/or `05_src/.env`:

- `API_GATEWAY_KEY` (for `app.py`)
- `WOLFRAM_APP_ID` (optional, improves math fact responses)

## Run

From repo root:

- Gateway app:
  - `"/Users/triptijoshi/Documents/Deploying_AI_course/deploying-ai/.venv/bin/python" "05_src/assignment_chat/app.py"`

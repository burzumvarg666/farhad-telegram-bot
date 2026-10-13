# Farhad Telegram AI Bot

A Persian/English general-purpose AI assistant built with FastAPI and deployed on Vercel.

## Current features

- Telegram webhook at `/api/telegram`
- Persistent reply-keyboard menu for help, status, memory status, and bot information
- Conversational assistant personality: professional, direct, and lightly witty
- Image input and analysis through the existing vision route
- Multi-provider routing and fallback using the existing environment configuration
- Temporary per-chat conversation memory (up to 14 messages, expires after 24 hours)
- Health endpoint at `/health`

## Commands

- `/start` - welcome message and interactive menu
- `/help` - command list and menu
- `/status` - service and model-routing summary
- `/memory` - temporary memory status
- `/reset` - clear this chat's temporary memory
- `/about` - bot information

## Environment variables

Configure secrets in Vercel Project Settings. Never commit API keys or bot tokens.

- `TELEGRAM_BOT_TOKEN` (or legacy `TELEGRAM_TOKEN`)
- `TOKENHARBOR_API_KEY`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- Optional model overrides: `GEMINI_MODEL`, `DEEPSEEK_FAST_MODEL`, `DEEPSEEK_MODEL`, `GROQ_MODEL`, `GROQ_FALLBACK`

## Development and tests

Install dependencies from `requirements.txt`, then run:

```bash
python -m unittest discover -s tests -v
```

The current memory is process-local and may be lost when a Vercel instance restarts. Persistent memory and games/scorekeeping are planned as later phases; they are not yet implemented.

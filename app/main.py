import os
import time
import base64
import mimetypes
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

app = FastAPI(title="Farhad Telegram AI Bot")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
TOKENHARBOR_KEY = os.getenv("TOKENHARBOR_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEEPSEEK_FAST_MODEL = os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash:free")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash:free")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK = os.getenv("GROQ_FALLBACK", "qwen/qwen3.8-27b")

MEMORY_TTL = 24 * 60 * 60
MAX_MESSAGES = 14
memory: dict[int, dict[str, Any]] = {}

SYSTEM_PROMPT = (
    "You are Farhad's general-purpose AI assistant. Answer in the user's language. "
    "Your personality combines a capable professional assistant with a lightly witty, "
    "curious conversational partner. Use dry, occasional humor when it fits, but never "
    "mock the user, overdo jokes, or let humor obscure the answer. Be direct, practical, "
    "accurate, and concise when possible. Explain technical subjects clearly. "
    "Do not assume the user is asking about paint/coatings unless the message says so. "
    "Never claim to remember personal facts that are not present in the conversation."
)


def main_menu() -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "گفتگو با هوش مصنوعی", "callback_data": "menu:ask"},
                {"text": "وضعیت حافظه", "callback_data": "menu:memory"},
            ],
            [
                {"text": "راهنما", "callback_data": "menu:help"},
                {"text": "وضعیت سرویس", "callback_data": "menu:status"},
            ],
            [
                {"text": "درباره ربات", "callback_data": "menu:about"},
            ],
        ]
    }


def remember(chat_id: int, role: str, content: str) -> None:
    now = time.time()
    item = memory.get(chat_id)
    if not item or now - item["updated"] > MEMORY_TTL:
        item = {"messages": [], "updated": now}
        memory[chat_id] = item
    item["messages"].append({"role": role, "content": content})
    item["messages"] = item["messages"][-MAX_MESSAGES:]
    item["updated"] = now


def history(chat_id: int) -> list[dict[str, str]]:
    item = memory.get(chat_id)
    if not item or time.time() - item["updated"] > MEMORY_TTL:
        return []
    return item["messages"][-MAX_MESSAGES:]


def reset_memory(chat_id: int) -> None:
    memory.pop(chat_id, None)


def route(text: str, has_image: bool = False) -> list[str]:
    # One primary provider per request. Fallbacks are used only after a real failure.
    if has_image:
        return ["deepseek", "groq"]

    t = text.lower()
    complex_terms = [
        "analyze", "analysis", "reason", "reasoning", "debug", "code", "python",
        "program", "architecture", "algorithm", "technical", "compare", "research",
        "long", "تحلیل", "بررسی", "استدلال", "کدنویسی", "پایتون", "برنامه", "معماری",
        "الگوریتم", "فنی", "مقایسه", "تحقیق", "محاسبه", "اکسل", "origin", "api",
    ]
    if len(text) > 900 or any(x in t for x in complex_terms):
        return ["deepseek", "groq"]
    return ["deepseek_fast", "gemini", "groq"]


async def openai_chat(
    base_url: str,
    key: str,
    model: str,
    messages: list[dict[str, Any]],
) -> str:
    client = AsyncOpenAI(api_key=key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=2500,
        timeout=15.0,
    )
    return (response.choices[0].message.content or "").strip()


async def gemini(text: str, msgs: list[dict[str, Any]]) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("Gemini is not configured")

    contents = []
    for m in msgs:
        contents.append({
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        })
    contents.append({"role": "user", "parts": [{"text": text}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2500},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def answer_text(
    chat_id: int,
    text: str,
    image_url: str | None = None,
) -> tuple[str, str]:
    old = history(chat_id)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(old)

    if image_url:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text or "Analyze this image carefully and answer in Persian.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        })
    else:
        messages.append({"role": "user", "content": text})

    for provider in route(text, bool(image_url)):
        try:
            if provider == "deepseek_fast" and TOKENHARBOR_KEY:
                out = await openai_chat(
                    "https://tokenharbor.ai/v1",
                    TOKENHARBOR_KEY,
                    DEEPSEEK_FAST_MODEL,
                    messages,
                )
            elif provider == "deepseek" and TOKENHARBOR_KEY:
                out = await openai_chat(
                    "https://tokenharbor.ai/v1",
                    TOKENHARBOR_KEY,
                    DEEPSEEK_MODEL,
                    messages,
                )
            elif provider == "groq" and GROQ_KEY:
                out = await openai_chat(
                    "https://api.groq.com/openai/v1",
                    GROQ_KEY,
                    GROQ_MODEL,
                    messages,
                )
                if not out and GROQ_FALLBACK:
                    out = await openai_chat(
                        "https://api.groq.com/openai/v1",
                        GROQ_KEY,
                        GROQ_FALLBACK,
                        messages,
                    )
            elif provider == "gemini" and not image_url and GEMINI_KEY:
                out = await gemini(text, old)
            else:
                continue

            if out:
                return out, provider
        except Exception:
            continue

    raise RuntimeError("No AI provider is currently available")


async def telegram(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Telegram token is not configured")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def download_telegram_image(file_path: str) -> str:
    url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.content

    mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def send_long(chat_id: int, text: str) -> None:
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]
    for chunk in chunks:
        await telegram("sendMessage", {"chat_id": chat_id, "text": chunk})


async def send_menu(chat_id: int, text: str) -> None:
    await telegram(
        "sendMessage",
        {"chat_id": chat_id, "text": text, "reply_markup": main_menu()},
    )


async def handle_callback(callback: dict[str, Any]) -> None:
    callback_id = callback.get("id")
    message = callback.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    action = callback.get("data", "")

    if callback_id:
        try:
            await telegram("answerCallbackQuery", {"callback_query_id": callback_id})
        except Exception:
            pass
    if chat_id is None:
        return

    if action == "menu:ask":
        await send_long(chat_id, "پیامت رو همین‌جا بنویس؛ اگر لازم باشه سؤال تکمیلی می‌پرسم، نه اینکه وانمود کنم همه‌چیز رو از غیب می‌دونم.")
    elif action == "menu:memory":
        await send_long(
            chat_id,
            f"در حافظه موقت این گفتگو {len(history(int(chat_id)))} پیام از حداکثر {MAX_MESSAGES} پیام ثبت شده. "
            "این حافظه فعلاً موقتی است و با راه‌اندازی مجدد سرویس ممکن است پاک شود.",
        )
    elif action == "menu:help":
        await send_menu(
            int(chat_id),
            "راهنمای سریع\n\n"
            "/start نمایش منو\n/help راهنما\n/status وضعیت سرویس\n"
            "/reset پاک کردن حافظه موقت گفتگو\n/memory وضعیت حافظه\n/about درباره ربات\n\n"
            "متن یا عکس بفرست تا بررسی کنم.",
        )
    elif action == "menu:status":
        await send_long(
            int(chat_id),
            "سرویس وب فعال است. درخواست‌های عادی ابتدا به DeepSeek V4 Flash و درخواست‌های پیچیده یا تصویری به DeepSeek V4.1 Flash می‌روند؛ Gemini و Groq مسیرهای جایگزین‌اند.",
        )
    elif action == "menu:about":
        await send_menu(
            int(chat_id),
            "Farhad Telegram AI Bot\n"
            "دستیار عمومی با لحن حرفه‌ای، کمی شوخ‌طبعی و حافظه موقت گفتگو. "
            "حافظه دائمی و بازی‌ها در مرحله‌های بعدی توسعه اضافه می‌شوند.",
        )


@app.get("/")
async def root():
    return {"ok": True, "service": "farhad-telegram-bot"}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "telegram_configured": bool(TELEGRAM_TOKEN),
        "ai_configured": bool(GEMINI_KEY or GROQ_KEY or TOKENHARBOR_KEY),
        "provider": "fast-router",
        "routing": {
            "normal": "deepseek-v4-flash -> gemini -> groq",
            "complex": "deepseek-v4.1-flash -> groq",
            "image": "deepseek-v4.1-flash -> groq",
        },
        "timeouts_seconds": 15,
        "gemini_model": GEMINI_MODEL,
        "deepseek_fast_model": DEEPSEEK_FAST_MODEL,
        "deepseek_model": DEEPSEEK_MODEL,
        "groq_model": GROQ_MODEL,
        "groq_fallback": GROQ_FALLBACK,
        "memory_backend": "session",
        "memory_max_messages": MAX_MESSAGES,
        "interactive_menu": True,
    }


@app.post("/api/telegram")
async def telegram_webhook(request: Request):
    update = await request.json()

    callback = update.get("callback_query")
    if callback:
        await handle_callback(callback)
        return {"ok": True}

    message = update.get("message") or update.get("edited_message")
    if not message or "chat" not in message:
        return JSONResponse({"ok": True})

    chat_id = message["chat"]["id"]
    text = (message.get("text") or message.get("caption") or "").strip()

    if text.startswith("/start"):
        await send_menu(
            chat_id,
            "سلام. من دستیار هوش مصنوعی فارهاد هستم.\n\n"
            "می‌تونم به سؤال‌ها جواب بدم، متن و کد رو بررسی کنم و عکس‌ها رو تحلیل کنم. "
            "لحنم حرفه‌ایه، ولی قرار نیست شبیه دفترچه راهنمای یخچال حرف بزنم.\n\n"
            "از منو استفاده کن یا مستقیم پیام بده.",
        )
        return {"ok": True}
    if text.startswith("/help"):
        await send_menu(
            chat_id,
            "/start شروع و نمایش منو\n/help راهنما\n/status وضعیت سرویس\n"
            "/reset پاک کردن حافظه گفتگو\n/memory وضعیت حافظه\n/about درباره ربات\n\n"
            "متن یا عکس بفرست تا شروع کنیم.",
        )
        return {"ok": True}
    if text.startswith("/reset"):
        reset_memory(chat_id)
        await send_long(chat_id, "حافظه موقت این گفتگو پاک شد.")
        return {"ok": True}
    if text.startswith("/memory"):
        await send_long(
            chat_id,
            f"تعداد پیام‌های ذخیره‌شده: {len(history(chat_id))} از {MAX_MESSAGES}. "
            "این حافظه موقت است و هنوز در پایگاه داده دائمی ذخیره نمی‌شود.",
        )
        return {"ok": True}
    if text.startswith("/status"):
        await send_long(
            chat_id,
            "سرویس فعال است. درخواست‌های عادی ابتدا به DeepSeek V4 Flash، "
            "و درخواست‌های پیچیده/تصویری به DeepSeek V4.1 Flash می‌روند. "
            "Gemini و Groq فقط fallback هستند.",
        )
        return {"ok": True}
    if text.startswith("/about"):
        await send_menu(
            chat_id,
            "Farhad Telegram AI Bot\n"
            "دستیار عمومی با حافظه موقت و چند مسیر جایگزین هوش مصنوعی. "
            "همه مدل‌های فعلی طبق تنظیمات موجود حفظ شده‌اند.",
        )
        return {"ok": True}

    image_url = None
    if message.get("photo"):
        try:
            file_id = message["photo"][-1]["file_id"]
            info = await telegram("getFile", {"file_id": file_id})
            file_path = info["result"]["file_path"]
            image_url = await download_telegram_image(file_path)
        except Exception:
            image_url = None

    if not text and not image_url:
        return {"ok": True}

    try:
        out, _provider = await answer_text(chat_id, text, image_url)
        remember(chat_id, "user", text or "[image]")
        remember(chat_id, "assistant", out)
        await send_long(chat_id, out)
    except Exception:
        await send_long(
            chat_id,
            "فعلاً هیچ مسیر رایگان هوش مصنوعی پاسخ‌گو نیست. چند لحظه بعد دوباره امتحان کن.",
        )

    return {"ok": True}

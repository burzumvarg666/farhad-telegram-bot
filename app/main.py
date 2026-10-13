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
TOKENHARBOR_KEY = os.getenv("TOKENHARBOR_API_KEY") or os.getenv("ZAI_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEEPSEEK_FAST_MODEL = os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash:free")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash:free")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK = os.getenv("GROQ_FALLBACK", "qwen/qwen3.8-27b")

MEMORY_TTL = 24 * 60 * 60
MAX_MESSAGES = 14
memory: dict[int, dict[str, Any]] = {}
game_states: dict[int, dict[str, Any]] = {}
game_scores: dict[int, int] = {}
QUIZ_QUESTIONS = [
    {"question": "بزرگ‌ترین سیاره منظومه شمسی کدام است؟", "options": ["زمین", "مشتری", "مریخ", "زهره"], "answer": "مشتری"},
    {"question": "واحد اندازه‌گیری مقاومت الکتریکی چیست؟", "options": ["ولت", "آمپر", "اهم", "وات"], "answer": "اهم"},
    {"question": "کدام زبان برنامه‌نویسی معمولاً برای تحلیل داده استفاده می‌شود؟", "options": ["Python", "HTML", "CSS", "SQL فقط"], "answer": "Python"},
]

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
    # Reply-keyboard buttons arrive as normal Telegram messages, so this menu
    # does not depend on callback_query being enabled in the webhook.
    return {
        "keyboard": [
            [
                {"text": "گفتگو با هوش مصنوعی"},
                {"text": "وضعیت حافظه"},
            ],
            [
                {"text": "راهنما"},
                {"text": "وضعیت سرویس"},
            ],
            [
                {"text": "درباره ربات"},
                {"text": "بازی و چالش"},
            ],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "input_field_placeholder": "پیامت رو بنویس...",
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


async def _supabase_request(method: str, table: str, *, params=None, payload=None):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=representation"}
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.request(method, f"{SUPABASE_URL}/rest/v1/{table}", params=params, json=payload, headers=headers)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return []
        return response.json()


async def restore_memory(chat_id: int) -> bool:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False
    try:
        rows = await _supabase_request("GET", "bot_memory", params={"chat_id": f"eq.{chat_id}", "select": "role,content,created_at", "order": "created_at.desc", "limit": str(MAX_MESSAGES)})
        if rows is None:
            return False
        memory[chat_id] = {"messages": [{"role": row["role"], "content": row["content"]} for row in reversed(rows)], "updated": time.time()}
        return True
    except Exception:
        return False


async def remember_durable(chat_id: int, role: str, content: str) -> None:
    remember(chat_id, role, content)
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            await _supabase_request("POST", "bot_memory", payload={"chat_id": chat_id, "role": role, "content": content})
        except Exception:
            pass


async def reset_durable_memory(chat_id: int) -> None:
    reset_memory(chat_id)
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            await _supabase_request("DELETE", "bot_memory", params={"chat_id": f"eq.{chat_id}"})
        except Exception:
            pass


async def get_game_score(chat_id: int) -> int:
    if chat_id in game_scores:
        return game_scores[chat_id]
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            rows = await _supabase_request("GET", "bot_game_scores", params={"chat_id": f"eq.{chat_id}", "select": "score", "limit": "1"})
            if rows:
                game_scores[chat_id] = int(rows[0]["score"])
        except Exception:
            pass
    return game_scores.get(chat_id, 0)


async def award_game_point(chat_id: int) -> int:
    score = await get_game_score(chat_id) + 1
    game_scores[chat_id] = score
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            await _supabase_request("POST", "bot_game_scores", params={"on_conflict": "chat_id"}, payload={"chat_id": chat_id, "score": score})
        except Exception:
            pass
    return score


async def send_keyboard(chat_id: int, text: str, rows: list[list[str]]) -> None:
    keyboard = [[{"text": label} for label in row] for row in rows]
    await telegram("sendMessage", {"chat_id": chat_id, "text": text, "reply_markup": {"keyboard": keyboard, "resize_keyboard": True, "is_persistent": True}})


async def start_quiz(chat_id: int) -> None:
    game_states[chat_id] = {"mode": "quiz", "index": 0, "round_score": 0}
    q = QUIZ_QUESTIONS[0]
    await send_keyboard(chat_id, "مسابقه اطلاعات عمومی شروع شد. هر پاسخ درست یک امتیاز دارد.\n\n" + q["question"], [q["options"][:2], q["options"][2:], ["خروج از بازی", "بازگشت به منو"]])


async def start_riddle(chat_id: int) -> None:
    game_states[chat_id] = {"mode": "riddle"}
    await send_keyboard(chat_id, "معمای کلمه: دندان دارم، اما غذا نمی‌خورم؛ موها را مرتب می‌کنم. من چی‌ام؟", [["خروج از بازی", "بازگشت به منو"]])


async def handle_game_message(chat_id: int, text: str) -> bool:
    if text in {"بازی و چالش", "/games"}:
        await send_keyboard(chat_id, f"بازی‌ها و چالش‌ها\nامتیاز ثبت‌شده شما: {await get_game_score(chat_id)}\nیک بازی انتخاب کن:", [["مسابقه اطلاعات عمومی", "معمای کلمه"], ["بازگشت به منو"]])
        return True
    if text == "بازگشت به منو":
        game_states.pop(chat_id, None)
        await send_menu(chat_id, "منوی اصلی آماده است.")
        return True
    if text in {"مسابقه اطلاعات عمومی", "/quiz"}:
        await start_quiz(chat_id)
        return True
    if text in {"معمای کلمه", "/riddle"}:
        await start_riddle(chat_id)
        return True
    state = game_states.get(chat_id)
    if not state:
        return False
    if text == "خروج از بازی":
        game_states.pop(chat_id, None)
        await send_menu(chat_id, f"بازی تمام شد. امتیاز کلی شما: {await get_game_score(chat_id)}")
        return True
    if state["mode"] == "riddle":
        if text.strip().casefold() in {"شانه", "شونه"}:
            score = await award_game_point(chat_id)
            await send_long(chat_id, f"درست گفتی. یک امتیاز گرفتی؛ امتیاز کل: {score}.")
        else:
            await send_long(chat_id, "این جواب نیست. پاسخ درست «شانه» است.")
        game_states.pop(chat_id, None)
        await send_menu(chat_id, "برای بازی بعدی از گزینه «بازی و چالش» استفاده کن.")
        return True
    if state["mode"] == "quiz":
        q = QUIZ_QUESTIONS[state["index"]]
        if text.casefold() == q["answer"].casefold():
            state["round_score"] += 1
            await send_long(chat_id, "درست است.")
        else:
            await send_long(chat_id, f"نه، پاسخ درست «{q['answer']}» بود.")
        state["index"] += 1
        if state["index"] >= len(QUIZ_QUESTIONS):
            for _ in range(state["round_score"]):
                await award_game_point(chat_id)
            total = await get_game_score(chat_id)
            result = state["round_score"]
            game_states.pop(chat_id, None)
            await send_menu(chat_id, f"مسابقه تمام شد. نتیجه این دور: {result} از {len(QUIZ_QUESTIONS)}؛ امتیاز کلی: {total}.")
            return True
        q = QUIZ_QUESTIONS[state["index"]]
        await send_keyboard(chat_id, f"سؤال {state['index'] + 1} از {len(QUIZ_QUESTIONS)}\n{q['question']}", [q["options"][:2], q["options"][2:], ["خروج از بازی", "بازگشت به منو"]])
        return True
    return False


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
        "memory_backend": "supabase" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "session",
        "memory_persistent_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        "memory_max_messages": MAX_MESSAGES,
        "games_enabled": True,
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

    # Reply-keyboard labels are ordinary messages; translate them into actions.
    menu_actions = {
        "وضعیت حافظه": "/memory",
        "راهنما": "/help",
        "وضعیت سرویس": "/status",
        "درباره ربات": "/about",
        "بازی و چالش": "/games",
    }
    if text == "گفتگو با هوش مصنوعی":
        await send_long(chat_id, "پیامت رو همین‌جا بنویس تا گفتگو رو ادامه بدیم.")
        return {"ok": True}
    text = menu_actions.get(text, text)

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
        await reset_durable_memory(chat_id)
        await send_long(chat_id, "حافظه این گفتگو پاک شد.")
        return {"ok": True}
    if text.startswith("/memory"):
        await restore_memory(chat_id)
        backend = "پایگاه داده دائمی" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "حافظه موقت سرور"
        await send_long(chat_id, f"تعداد پیام‌های ذخیره‌شده: {len(history(chat_id))} از {MAX_MESSAGES}. محل ذخیره: {backend}. " + ("" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "برای فعال‌شدن حافظه دائمی باید Supabase را طبق راهنمای پروژه وصل کنیم."))
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

    if await handle_game_message(chat_id, text):
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
        await restore_memory(chat_id)
        out, _provider = await answer_text(chat_id, text, image_url)
        await remember_durable(chat_id, "user", text or "[image]")
        await remember_durable(chat_id, "assistant", out)
        await send_long(chat_id, out)
    except Exception:
        await send_long(
            chat_id,
            "فعلاً هیچ مسیر رایگان هوش مصنوعی پاسخ‌گو نیست. چند لحظه بعد دوباره امتحان کن.",
        )

    return {"ok": True}

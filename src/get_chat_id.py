"""One-shot helper: poll getUpdates and print chat_id after user sends /start to bot."""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN", "")

print(f"Bot: @nebula_meteo_bot")
print("Open Telegram, find the bot, and send /start")
print("Waiting up to 60 seconds...\n")

for attempt in range(12):
    r = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        timeout=10,
    )
    updates = r.json().get("result", [])
    if updates:
        for u in updates:
            msg = u.get("message", {})
            chat = msg.get("chat", {})
            chat_id = chat.get("id")
            username = msg.get("from", {}).get("username", "—")
            text = msg.get("text", "")
            print(f"  Message : {text!r}")
            print(f"  chat_id : {chat_id}")
            print(f"  from    : @{username}")
            print(f"\nAdd to .env:  TELEGRAM_CHAT_ID={chat_id}")
        break
    print(f"  attempt {attempt + 1}/12 — no messages yet ...")
    time.sleep(5)
else:
    print("No messages received. Make sure you sent /start to the bot.")

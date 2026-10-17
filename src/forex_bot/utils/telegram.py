"""Telegram logging integration using only the Python standard library."""

from __future__ import annotations

import json
import logging
import os
import urllib.request


class TelegramLoggingHandler(logging.Handler):
    """Send warning-and-higher records to a Telegram chat when configured."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        super().__init__()
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.setFormatter(
            logging.Formatter(
                "🚨 *Trading Bot Alert*\n\n"
                "*Level:* %(levelname)s\n"
                "*Message:* %(message)s"
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        if not self.bot_token or not self.chat_id:
            return
        try:
            request = urllib.request.Request(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                data=json.dumps(
                    {
                        "chat_id": self.chat_id,
                        "text": self.format(record),
                        "parse_mode": "Markdown",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5):
                return
        except Exception:
            # Never interrupt trading, and avoid recursively logging notifier errors.
            self.handleError(record)

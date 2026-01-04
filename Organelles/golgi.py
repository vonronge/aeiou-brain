"""
AEIOU Brain — Local Multimodal AI Ecosystem

Copyright © 2026 Frederick von Rönge
GitHub: https://github.com/vonronge/aeiou-brain

The Golgi Apparatus:
Central dispatch for system messages and telemetry.
Packages logs and routes them to attached sinks (GUI, CLI, Files).
"""

import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class LogRecord:
    timestamp: str
    level: str  # INFO, WARN, ERROR, SUCCESS, SAVE, HARVEST
    message: str
    tag: Optional[str] = None
    source: str = "System"


class Organelle_Golgi:
    def __init__(self):
        self._sinks: Dict[str, Callable[[LogRecord], None]] = {}
        self._lock = threading.Lock()

        # Default Sink: Print to Console (for Headless safety)
        self.attach_sink("cli", self._default_cli_sink)

    def attach_sink(self, name: str, callback: Callable[[LogRecord], None]):
        """
        Connect a UI element or file logger.
        Callback signature: fn(record: LogRecord)
        """
        with self._lock:
            self._sinks[name] = callback

    def detach_sink(self, name: str):
        with self._lock:
            if name in self._sinks:
                del self._sinks[name]

    def _default_cli_sink(self, record: LogRecord):
        # Basic formatting for terminal
        print(f"[{record.timestamp}] [{record.level}] {record.message}")

    def _dispatch(self, level: str, message: str, tag: str = None, source: str = "System"):
        ts = datetime.now().strftime("%H:%M:%S")
        record = LogRecord(timestamp=ts, level=level, message=message, tag=tag, source=source)

        # Snapshot sinks to avoid lock contention during I/O
        with self._lock:
            active_sinks = list(self._sinks.values())

        for sink in active_sinks:
            try:
                sink(record)
            except Exception:
                pass  # Sinks must not crash the kernel

    # --- PUBLIC API ---
    def info(self, msg, source="System"):
        self._dispatch("INFO", msg, source=source)

    def warn(self, msg, source="System"):
        self._dispatch("WARN", msg, source=source)

    def error(self, msg, source="System"):
        self._dispatch("ERROR", msg, source=source)

    def success(self, msg, source="System"):
        self._dispatch("SUCCESS", msg, source=source)

    def save(self, msg, source="System"):
        self._dispatch("SAVE", msg, source=source)

    def harvest(self, msg, source="System"):
        self._dispatch("HARVEST", msg, source=source)
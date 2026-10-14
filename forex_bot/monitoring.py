from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeTelemetry:
    last_updated: str = field(default_factory=_now_iso)
    last_equity: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    order_rejects: int = 0
    ws_reconnects: int = 0
    kill_switch_trips: int = 0
    execution_halts: int = 0
    pnl_drift_alerts: int = 0
    open_positions: int = 0
    recent_events: list[str] = field(default_factory=list)

    def add_event(self, event: str) -> None:
        self.recent_events.append(f"{_now_iso()} {event}")
        if len(self.recent_events) > 20:
            self.recent_events = self.recent_events[-20:]
        self.last_updated = _now_iso()


class TelemetryStore:
    def __init__(self, path: str | Path = "data/db/telemetry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RuntimeTelemetry:
        if not self.path.exists():
            return RuntimeTelemetry()
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return RuntimeTelemetry()
        return RuntimeTelemetry(**data)

    def save(self, telemetry: RuntimeTelemetry) -> None:
        telemetry.last_updated = _now_iso()
        self.path.write_text(json.dumps(telemetry.__dict__, indent=2, sort_keys=True))

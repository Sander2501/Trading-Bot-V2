from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import CapitalCredentials, TradingConfig


@dataclass
class StartupCheck:
    name: str
    ok: bool
    detail: str = ""


def run_startup_checks(
    config: TradingConfig,
    creds: CapitalCredentials,
    *,
    data_dir: str = "data/historical",
    max_data_age_hours: int = 48,
) -> list[StartupCheck]:
    out: list[StartupCheck] = []

    out.extend(_credentials_check(creds))
    out.extend(_config_check(config))
    out.extend(_clock_check())
    out.extend(_data_freshness_check(config, data_dir, max_data_age_hours))
    return out


def all_checks_passed(checks: list[StartupCheck]) -> bool:
    return bool(checks) and all(c.ok for c in checks)


def format_startup_checks(checks: list[StartupCheck]) -> str:
    lines = []
    passed = 0
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        if c.ok:
            passed += 1
        detail = f" — {c.detail}" if c.detail else ""
        lines.append(f"[{mark}] {c.name}{detail}")
    return f"Startup checks: {passed}/{len(checks)} passed\n" + "\n".join(f"  {x}" for x in lines)


def _credentials_check(creds: CapitalCredentials) -> list[StartupCheck]:
    checks: list[StartupCheck] = []
    env = creds.environment.lower()
    checks.append(StartupCheck("environment", env in {"demo", "live"}, f"environment={env}"))
    checks.append(StartupCheck("api_key_present", bool(creds.api_key), "api key loaded"))
    return checks


def _config_check(config: TradingConfig) -> list[StartupCheck]:
    checks: list[StartupCheck] = []
    checks.append(
        StartupCheck("instruments_configured", bool(config.instruments),
                     f"instruments={len(config.instruments)}")
    )
    checks.append(
        StartupCheck("strategy_configured", bool(config.strategy), f"strategy={config.strategy}")
    )
    if config.state_db:
        parent = Path(config.state_db).parent
        ok = True
        detail = f"path={parent}"
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            ok = False
            detail = f"path={parent} error={exc}"
        checks.append(StartupCheck("state_db_dir", ok, detail))
    else:
        checks.append(StartupCheck("state_db_dir", True, "in-memory state"))
    return checks


def _clock_check() -> list[StartupCheck]:
    now = datetime.now(timezone.utc)
    return [StartupCheck("clock_utc", True, f"utc={now.isoformat()}")]


def _data_freshness_check(
    config: TradingConfig, data_dir: str, max_data_age_hours: int
) -> list[StartupCheck]:
    checks: list[StartupCheck] = []
    root = Path(data_dir)
    now = datetime.now(timezone.utc).timestamp()
    max_age_seconds = max_data_age_hours * 3600
    for inst in config.instruments:
        files = list(root.glob(f"{inst.epic}__{inst.timeframe}.csv"))
        if not files:
            checks.append(StartupCheck(
                f"data_freshness:{inst.epic}", False,
                f"missing {inst.epic}/{inst.timeframe}; run download",
            ))
            continue
        latest = max(files, key=lambda f: f.stat().st_mtime)
        age = now - latest.stat().st_mtime
        checks.append(
            StartupCheck(
                f"data_freshness:{inst.epic}",
                age <= max_age_seconds,
                f"age_hours={age / 3600:.1f}",
            )
        )
    return checks

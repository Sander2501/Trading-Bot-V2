from pathlib import Path

from forex_bot.config import CapitalCredentials, InstrumentConfig, TradingConfig
from forex_bot.startup_checks import all_checks_passed, run_startup_checks
from tests.helpers import make_candles


def _creds():
    return CapitalCredentials("demo", "u", "p", "k")


def test_startup_checks_fail_without_data(tmp_path):
    cfg = TradingConfig(instruments=[InstrumentConfig("EURUSD", "MINUTE_15")])
    checks = run_startup_checks(cfg, _creds(), data_dir=str(tmp_path), max_data_age_hours=1)
    assert not all_checks_passed(checks)
    assert any(c.name.startswith("data_freshness") and not c.ok for c in checks)


def test_startup_checks_pass_with_recent_data(tmp_path):
    cfg = TradingConfig(instruments=[InstrumentConfig("EURUSD", "MINUTE_15")])
    candles = make_candles([1.1, 1.2, 1.3], epic="EURUSD", timeframe="MINUTE_15")
    out = Path(tmp_path) / "EURUSD__MINUTE_15.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
        for c in candles
    )
    out.write_text("\n".join(rows) + "\n")
    checks = run_startup_checks(cfg, _creds(), data_dir=str(tmp_path), max_data_age_hours=1000)
    assert all_checks_passed(checks)

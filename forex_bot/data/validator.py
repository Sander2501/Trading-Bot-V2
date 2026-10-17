"""Validation helpers for historical market data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from ..logging_setup import get_logger

log = get_logger(__name__)


def validate_candle_continuity(
    candles: Sequence[Mapping[str, Any]], timeframe_minutes: int
) -> bool:
    """Return whether candles contain no non-weekend gaps."""
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    if len(candles) < 2:
        return True
    expected_delta = timedelta(minutes=timeframe_minutes)
    for previous, current in zip(candles, candles[1:]):
        previous_time = previous["timestamp"]
        current_time = current["timestamp"]
        if not isinstance(previous_time, datetime) or not isinstance(current_time, datetime):
            raise TypeError("candle timestamps must be datetime objects")
        if previous_time.tzinfo is None or current_time.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        actual_delta = current_time - previous_time
        if actual_delta > expected_delta and not _is_weekend_gap(previous_time, current_time):
            log.error(
                "Data gap detected",
                extra={
                    "missing_duration": str(actual_delta),
                    "from_timestamp": previous_time.isoformat(),
                    "to_timestamp": current_time.isoformat(),
                },
            )
            return False
    log.info("Data continuity check passed")
    return True


def _is_weekend_gap(previous: datetime, current: datetime) -> bool:
    return previous.weekday() == 4 and current.weekday() in {5, 6, 0}

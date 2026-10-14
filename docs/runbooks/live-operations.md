# Live Operations Runbook

## 1) WebSocket disconnect / reconnect loop
- Check `forex-bot dashboard` for `ws reconnects` and recent events.
- If reconnects spike, pause trading and run:
  - `forex-bot validate-env --environment demo`
  - `forex-bot preflight --environment demo`
- Verify network stability and broker status before restarting `demo`/`live`.

## 2) Order reject spike
- Watch `order rejects` in dashboard and logs (`entry rejected by broker`).
- Common causes: spread expansion, insufficient margin, min deal-size mismatch.
- Actions:
  - Re-run `forex-bot spreads` and update instrument `spread_points`.
  - Reduce risk profile (`--profile live-safe`) and restart.
  - Re-run `preflight --test-order` on demo.

## 3) Kill-switch trip
- Dashboard and logs report `kill switch tripped`.
- The engine flattens positions and halts new entries by design.
- Required operator flow:
  1. Export latest logs and inspect drawdown root cause.
  2. Confirm broker positions are flat.
  3. Re-validate config/risk settings.
  4. Resume only after manual approval and `validate-env` + `preflight`.

## 4) Stale local/broker position mismatch
- Use `preflight` to confirm account and open positions.
- Restart engine to force reconciliation (`_reconcile_positions` path).
- If mismatch remains, treat broker state as source of truth and manually flatten.

## 5) Emergency flatten
- Stop the bot process first (prevents re-entry).
- Close remaining broker positions from platform/API.
- Confirm no open positions with `preflight`.
- Preserve state/logs before any restart.

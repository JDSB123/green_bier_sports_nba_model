# 🏀 NBA Predictions - Quick Run

**THE ONE COMMAND:**

```powershell
python scripts/run_slate.py
```

That's it. This command:
1. ✅ Checks Docker is running
2. ✅ Starts the stack if needed
3. ✅ Waits for API to be ready
4. ✅ Fetches predictions
5. ✅ Displays picks with fire ratings

---

## Examples

```powershell
# Today's full slate
python scripts/run_slate.py

# Tomorrow's slate
python scripts/run_slate.py --date tomorrow

# Specific date
python scripts/run_slate.py --date 2025-12-19

# Filter to specific team
python scripts/run_slate.py --matchup Lakers

# Specific team on specific date
python scripts/run_slate.py --date tomorrow --matchup "Celtics"
```

---

## Output Example

```
================================================================================
🏀 NBA PREDICTIONS - TODAY
================================================================================

📊 Found 5 game(s)

────────────────────────────────────────────────────────────────────────────────
🎯 Chicago Bulls @ Cleveland Cavaliers
⏰ 7:00 PM CST

  FULL GAME:
    📌 SPREAD: Cleveland -7.5  |  Edge: +2.3 pts  |  🔥🔥🔥🔥
    📌 TOTAL: UNDER 223.5  |  Edge: +3.1 pts  |  🔥🔥🔥
    📌 ML: Cleveland (-300)  |  Edge: +5.2%  |  🔥🔥🔥🔥🔥

  FIRST HALF:
    📌 1H SPREAD: Cleveland -4.0  |  Edge: +1.5 pts  |  🔥🔥🔥
```

---

## Fire Rating Guide

| Rating | Meaning |
|--------|---------|
| 🔥🔥🔥🔥🔥 | Strong play - high confidence + large edge |
| 🔥🔥🔥🔥 | Good play |
| 🔥🔥🔥 | Moderate play |
| 🔥🔥 | Marginal |
| 🔥 | Low confidence |

---

## Troubleshooting

**Docker not running:**
```
Start Docker Desktop
```

**API not loading:**
```powershell
docker compose logs strict-api
```

**Models missing:**
```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

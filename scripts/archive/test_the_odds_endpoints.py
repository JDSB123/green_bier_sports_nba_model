#!/usr/bin/env python3
"""
Lightweight test harness for TheOdds API NBA endpoints.

Usage (pwsh):
  $env:THE_ODDS_API_KEY = '<your_key>'
  python .\scripts\test_the_odds_endpoints.py

The script loads any `.env*` files found in the repo root and `scripts/`
directory (later files override earlier). It masks the API key when printing
URLs so secrets do not appear in logs.
"""
import os
import sys
import glob
from pathlib import Path
import urllib.parse

try:
    import requests
    from dotenv import load_dotenv
except Exception:
    print("Missing dependency: requests or python-dotenv. Install with `pip install requests python-dotenv` or `pip install -r requirements.txt`.")
    raise

# Load .env files found in repo root and the scripts directory (do NOT override existing env vars)
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
env_files = []
for p in sorted(ROOT.glob('.env*')):
    env_files.append(p)
for p in sorted(SCRIPTS.glob('.env*')):
    if p not in env_files:
        env_files.append(p)
if env_files:
    for ef in env_files:
        try:
            # Do not override environment variables already set in the runtime
            load_dotenv(ef, override=False)
        except Exception:
            pass

API_KEY = os.environ.get("THE_ODDS_API_KEY")
if not API_KEY:
    print("Error: THE_ODDS_API_KEY environment variable is not set (or not provided via .env files).")
    print("Set it in PowerShell with: $env:THE_ODDS_API_KEY = '<key>' or add it to a .env file.")
    sys.exit(2)

BASE = "https://api.the-odds-api.com/v4"
TIMEOUT = 20

def mask_url(url: str) -> str:
    if not url:
        return url
    try:
        p = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        qs_masked = [(k, '[REDACTED]' if k.lower() == 'apikey' else v) for (k, v) in qs]
        new_q = urllib.parse.urlencode(qs_masked)
        return urllib.parse.urlunparse((p.scheme, p.netloc, p.path, p.params, new_q, p.fragment))
    except Exception:
        return url

def get(path, params=None):
    url = f"{BASE}{path}"
    params = params or {}
    params.update({"apiKey": API_KEY})
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
    except Exception as e:
        return {"ok": False, "error": str(e), "status": None, "url": url}
    out = {"ok": True, "status": r.status_code, "url": r.url}
    hdrs = r.headers
    out["headers"] = {k: hdrs[k] for k in hdrs if k.lower().startswith("x-requests") or k.lower().startswith("x-ratelimit")}
    ctype = r.headers.get("Content-Type", "")
    if "application/json" in ctype or r.text.startswith("{") or r.text.startswith("["):
        try:
            out["json"] = r.json()
        except Exception as e:
            out["json_error"] = str(e)
            out["text"] = r.text[:1000]
    else:
        out["text"] = r.text[:1000]
    return out

def sample_check_sports(resp):
    if resp.get("status") != 200:
        return False, "HTTP %s" % resp.get("status")
    j = resp.get("json")
    if not isinstance(j, list):
        return False, "expected array"
    for s in j:
        if s.get("key") == "basketball_nba":
            return True, "found basketball_nba"
    return False, "basketball_nba not found"

def sample_check_odds(resp):
    if resp.get("status") != 200:
        return False, "HTTP %s" % resp.get("status")
    j = resp.get("json")
    if isinstance(j, list):
        if not j:
            return False, "empty events list"
        ev = j[0]
        required = ["sport_key", "commence_time", "home_team", "away_team", "bookmakers"]
        missing = [k for k in required if k not in ev]
        if missing:
            return False, "missing fields: %s" % ",".join(missing)
        if ev.get("sport_key") != "basketball_nba":
            return False, f"unexpected sport_key {ev.get('sport_key')}"
        return True, f"events={len(j)}; sample event id={ev.get('id')}"
    elif isinstance(j, dict):
        return True, "dict payload"
    else:
        return False, "unexpected payload type"

def main():
    print("Testing TheOdds NBA endpoints (non-destructive).")
    summary = []

    print('\n1) GET /v4/sports')
    r = get("/sports")
    ok, msg = sample_check_sports(r) if r.get("ok") else (False, r.get("error"))
    print(f"  status={r.get('status')} url={mask_url(r.get('url'))}")
    print(f"  check: {ok} - {msg}")
    summary.append(("sports", ok, msg, r))

    print('\n2) GET /v4/sports/basketball_nba/events')
    r_events = get("/sports/basketball_nba/events", params={"dateFormat": "iso"})
    print(f"  status={r_events.get('status')} url={mask_url(r_events.get('url'))}")
    e_ok = False
    first_event_id = None
    if r_events.get("status") == 200 and isinstance(r_events.get("json"), list) and r_events.get("json"):
        first = r_events.get("json")[0]
        first_event_id = first.get("id") or first.get("event_id")
        e_ok = True
        print(f"  events returned: {len(r_events.get('json'))}, sample id={first_event_id}")
    else:
        print("  events check failed or empty")
    summary.append(("events", e_ok, f"first_id={first_event_id}", r_events))

    print('\n3) GET /v4/sports/basketball_nba/odds (regions=us, markets=h2h,spreads,totals)')
    r_odds = get("/sports/basketball_nba/odds", params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "dateFormat": "iso"})
    print(f"  status={r_odds.get('status')} url={mask_url(r_odds.get('url'))}")
    o_ok, o_msg = sample_check_odds(r_odds) if r_odds.get("ok") else (False, r_odds.get("error"))
    print(f"  check: {o_ok} - {o_msg}")
    summary.append(("odds", o_ok, o_msg, r_odds))

    if first_event_id:
        print(f"\n4) GET /v4/sports/basketball_nba/events/{first_event_id}/odds")
        r_event_odds = get(f"/sports/basketball_nba/events/{first_event_id}/odds", params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "dateFormat": "iso"})
        print(f"  status={r_event_odds.get('status')} url={mask_url(r_event_odds.get('url'))}")
        if r_event_odds.get("status") == 200:
            print("  returned single event object; checking for 'bookmakers'... ")
            j = r_event_odds.get("json")
            if isinstance(j, dict) and j.get("bookmakers"):
                print(f"  ok: bookmakers={len(j.get('bookmakers'))}")
                summary.append(("event_odds", True, f"bookmakers={len(j.get('bookmakers'))}", r_event_odds))
            else:
                print("  event odds returned but 'bookmakers' missing or empty")
                summary.append(("event_odds", False, "no_bookmakers", r_event_odds))
        else:
            print("  failed to fetch event odds")
            summary.append(("event_odds", False, f"status={r_event_odds.get('status')}", r_event_odds))
    else:
        print('\n4) Skipping event-level odds because no event id was found from events endpoint.')

    if first_event_id:
        print(f"\n5) GET /v4/sports/basketball_nba/events/{first_event_id}/markets")
        r_markets = get(f"/sports/basketball_nba/events/{first_event_id}/markets", params={"regions": "us", "dateFormat": "iso"})
        print(f"  status={r_markets.get('status')} url={mask_url(r_markets.get('url'))}")
        if r_markets.get("status") == 200:
            jm = r_markets.get("json")
            print(f"  markets payload type: {type(jm)}")
            summary.append(("markets", True, f"type={type(jm)}", r_markets))
        else:
            summary.append(("markets", False, f"status={r_markets.get('status')}", r_markets))

    print('\n6) GET /v4/sports/basketball_nba/scores (daysFrom=1)')
    r_scores = get("/sports/basketball_nba/scores", params={"daysFrom": "1", "dateFormat": "iso"})
    print(f"  status={r_scores.get('status')} url={mask_url(r_scores.get('url'))}")
    if r_scores.get("status") == 200:
        j = r_scores.get("json")
        print(f"  scores returned: type={type(j)}; count={(len(j) if isinstance(j, list) else 'n/a')}")
        summary.append(("scores", True, f"count={(len(j) if isinstance(j, list) else 'n/a')}", r_scores))
    else:
        summary.append(("scores", False, f"status={r_scores.get('status')}", r_scores))

    print('\n7) GET /v4/historical/sports/basketball_nba/odds (snapshot) - may require paid plan')
    hist_date = "2025-12-01T12:00:00Z"
    r_hist = get("/historical/sports/basketball_nba/odds", params={"regions": "us", "markets": "h2h,spreads,totals", "date": hist_date, "oddsFormat": "american", "dateFormat": "iso"})
    print(f"  status={r_hist.get('status')} url={mask_url(r_hist.get('url'))}")
    if r_hist.get("status") == 200:
        print("  historical snapshot returned (paid plan)")
        summary.append(("historical_odds", True, "ok", r_hist))
    elif r_hist.get("status") == 403:
        print("  403 Forbidden — historical access not enabled for this API key/account")
        summary.append(("historical_odds", False, "403_forbidden", r_hist))
    else:
        summary.append(("historical_odds", False, f"status={r_hist.get('status')}", r_hist))

    print('\n8) GET /v4/sports/basketball_nba/participants')
    r_parts = get("/sports/basketball_nba/participants")
    print(f"  status={r_parts.get('status')} url={mask_url(r_parts.get('url'))}")
    if r_parts.get("status") == 200 and isinstance(r_parts.get("json"), list):
        print(f"  participants returned: {len(r_parts.get('json'))}")
        summary.append(("participants", True, f"count={len(r_parts.get('json'))}", r_parts))
    else:
        summary.append(("participants", False, f"status={r_parts.get('status')}", r_parts))

    print('\n9) GET /v4/historical/sports/basketball_nba/events (paid plan)')
    r_hist_events = get("/historical/sports/basketball_nba/events", params={"date": hist_date})
    print(f"  status={r_hist_events.get('status')} url={mask_url(r_hist_events.get('url'))}")
    hist_event_id = None
    if r_hist_events.get("status") == 200 and isinstance(r_hist_events.get("json"), dict):
        data = r_hist_events.get("json").get("data")
        if isinstance(data, list) and data:
            hist_event_id = data[0].get("id")
            print(f"  historical events returned: {len(data)}, sample id={hist_event_id}")
            summary.append(("historical_events", True, f"count={len(data)}", r_hist_events))
        else:
            summary.append(("historical_events", False, "no_data", r_hist_events))
    elif r_hist_events.get("status") == 403:
        print("  403 Forbidden — historical events access not enabled for this API key/account")
        summary.append(("historical_events", False, "403_forbidden", r_hist_events))
    else:
        summary.append(("historical_events", False, f"status={r_hist_events.get('status')}", r_hist_events))

    if hist_event_id:
        print(f"\n10) GET /v4/historical/sports/basketball_nba/events/{hist_event_id}/odds (paid plan)")
        r_hist_event_odds = get(f"/historical/sports/basketball_nba/events/{hist_event_id}/odds", params={"date": hist_date, "regions": "us", "markets": "h2h,spreads,totals"})
        print(f"  status={r_hist_event_odds.get('status')} url={mask_url(r_hist_event_odds.get('url'))}")
        if r_hist_event_odds.get("status") == 200:
            print("  historical event odds returned")
            summary.append(("historical_event_odds", True, "ok", r_hist_event_odds))
        else:
            summary.append(("historical_event_odds", False, f"status={r_hist_event_odds.get('status')}", r_hist_event_odds))
    else:
        print('\n10) Skipping historical event odds because no historical event id was returned')

    print('\n11) GET /v4/sports/upcoming/odds (cross-sport upcoming events)')
    r_upcoming = get("/sports/upcoming/odds", params={"regions": "us", "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"})
    print(f"  status={r_upcoming.get('status')} url={mask_url(r_upcoming.get('url'))}")
    if r_upcoming.get("status") == 200 and isinstance(r_upcoming.get("json"), list):
        print(f"  upcoming events returned: {len(r_upcoming.get('json'))}")
        summary.append(("upcoming", True, f"count={len(r_upcoming.get('json'))}", r_upcoming))
    else:
        summary.append(("upcoming", False, f"status={r_upcoming.get('status')}", r_upcoming))

    print('\n\nSUMMARY:')
    for name, ok, msg, full in summary:
        print(f" - {name}: {'OK' if ok else 'FAIL'} - {msg} (status={full.get('status')})")

    print('\nNotes:')
    print(" - Use environment variable THE_ODDS_API_KEY to avoid leaking keys in logs.")
    print(" - Historical endpoints often require a paid plan and will return 403 if not enabled.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run TheOdds v4 endpoints for NBA games scheduled tomorrow (UTC).

This script finds `basketball_nba` events whose `commence_time` falls on
tomorrow's UTC date, then calls event-level endpoints for each event:
 - /sports/basketball_nba/events
 - /sports/basketball_nba/events/{eventId}/odds
 - /sports/basketball_nba/events/{eventId}/markets
Additionally it prints a short status summary for each call and masks the
`apiKey` query param in printed URLs.

Usage (PowerShell):
  $env:THE_ODDS_API_KEY = '<key>'
  python .\scripts\run_the_odds_tomorrow.py
  Remove-Item Env:\THE_ODDS_API_KEY
"""
import os
import sys
from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo
import urllib.parse

try:
    import requests
    from dotenv import load_dotenv
except Exception:
    print("Missing dependency: requests or python-dotenv. Install with pip install requests python-dotenv")
    raise

# Load .env files if present (do not override existing env vars)
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in sorted(ROOT.glob('.env*')):
    load_dotenv(p, override=False)
for p in sorted(SCRIPTS.glob('.env*')):
    load_dotenv(p, override=False)

API_KEY = os.environ.get('THE_ODDS_API_KEY')
if not API_KEY:
    print('Error: THE_ODDS_API_KEY not set (env or .env).')
    sys.exit(2)

BASE = 'https://api.the-odds-api.com/v4'
TIMEOUT = 20

def ensure_output_dir(target_date: str):
    """Create and return output directory for the target date."""
    out_dir = Path('data/raw/the_odds') / target_date
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def timestamp() -> str:
    """Return current UTC timestamp in compact format."""
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_json(data, filename: str, out_dir):
    """Save data to JSON file with pretty formatting."""
    import json
    filepath = out_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

def extract_rate_limit_info(headers) -> dict:
    """Extract rate limit headers from API response."""
    return {
        'requests_remaining': headers.get('x-requests-remaining'),
        'requests_used': headers.get('x-requests-used'),
        'timestamp': dt.datetime.utcnow().isoformat() + 'Z'
    }

def mask_url(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        qs_masked = [(k, '[REDACTED]' if k.lower() == 'apikey' else v) for k, v in qs]
        return urllib.parse.urlunparse((p.scheme, p.netloc, p.path, p.params, urllib.parse.urlencode(qs_masked), p.fragment))
    except Exception:
        return url

def get(path, params=None):
    url = f"{BASE}{path}"
    params = params or {}
    params.update({'apiKey': API_KEY})
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
    except Exception as e:
        return {'ok': False, 'error': str(e), 'status': None, 'url': url, 'headers': {}}
    out = {'ok': True, 'status': r.status_code, 'url': r.url, 'headers': dict(r.headers)}
    ctype = r.headers.get('Content-Type', '')
    if 'application/json' in ctype or r.text.startswith('{') or r.text.startswith('['):
        try:
            out['json'] = r.json()
        except Exception:
            out['text'] = r.text[:1000]
    else:
        out['text'] = r.text[:1000]
    return out

def tz_tomorrow_date(tz_name: str):
    tz = ZoneInfo(tz_name)
    now = dt.datetime.now(tz)
    tomorrow = now + dt.timedelta(days=1)
    return tomorrow.date().isoformat()

def is_on_date_iso(iso_ts: str, date_str: str, tz_name: str) -> bool:
    try:
        tz = ZoneInfo(tz_name)
        t = dt.datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        t_local = t.astimezone(tz)
        return t_local.date().isoformat() == date_str
    except Exception:
        return False


def to_tz_string(iso_ts: str, tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name)
        t = dt.datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        return t.astimezone(tz).isoformat()
    except Exception:
        return iso_ts


def normalize_name(s: str) -> str:
    if not s:
        return ''
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

def main():
    tz_name = 'America/Chicago'  # CST/CDT depending on date
    target = tz_tomorrow_date(tz_name)
    out_dir = ensure_output_dir(target)

    print(f"Fetching ALL The Odds API endpoints for NBA games on: {target} ({tz_name})")
    print(f"Output directory: {out_dir}")

    files_saved = []
    rate_limits = []

    # === STEP 1: Fetch sport-level endpoints ===
    print("\n=== Fetching sport-level endpoints ===")

    # 1a. Events list
    print("Fetching events list...")
    r_events = get('/sports/basketball_nba/events', params={'dateFormat': 'iso'})
    if r_events.get('status') == 200:
        filepath = save_json(r_events['json'], f'events_{timestamp()}.json', out_dir)
        files_saved.append(('events', filepath))
        rate_limits.append(extract_rate_limit_info(r_events['headers']))
        print(f"  [OK] Events: {len(r_events['json'])} total events saved to {filepath.name}")
    else:
        print(f"  [FAIL] Failed to fetch events: {r_events.get('status')}")
        sys.exit(1)

    # 1b. Sport-level odds (all games with bookmakers)
    print("Fetching sport-level odds...")
    r_sport_odds = get('/sports/basketball_nba/odds',
                       params={'regions': 'us', 'markets': 'h2h,spreads,totals',
                               'oddsFormat': 'american', 'dateFormat': 'iso'})
    if r_sport_odds.get('status') == 200:
        filepath = save_json(r_sport_odds['json'], f'sport_odds_{timestamp()}.json', out_dir)
        files_saved.append(('sport_odds', filepath))
        rate_limits.append(extract_rate_limit_info(r_sport_odds['headers']))
        print(f"  [OK] Sport odds: {len(r_sport_odds['json'])} events with odds saved to {filepath.name}")
    else:
        print(f"  [FAIL] Failed to fetch sport odds: {r_sport_odds.get('status')}")

    # 1c. Participants
    print("Fetching participants...")
    r_participants = get('/sports/basketball_nba/participants')
    participant_names = set()
    if r_participants.get('status') == 200:
        filepath = save_json(r_participants['json'], f'participants_{timestamp()}.json', out_dir)
        files_saved.append(('participants', filepath))
        rate_limits.append(extract_rate_limit_info(r_participants['headers']))
        # Build participant name set for consistency checking
        if isinstance(r_participants.get('json'), list):
            for p in r_participants.get('json'):
                participant_names.add(normalize_name(p.get('name') or p.get('team') or p.get('id')))
        print(f"  [OK] Participants: {len(r_participants['json'])} teams saved to {filepath.name}")
    else:
        print(f"  [FAIL] Failed to fetch participants: {r_participants.get('status')}")

    # 1d. Scores
    print("Fetching scores...")
    r_scores = get('/sports/basketball_nba/scores', params={'daysFrom': '1', 'dateFormat': 'iso'})
    if r_scores.get('status') == 200:
        filepath = save_json(r_scores['json'], f'scores_{timestamp()}.json', out_dir)
        files_saved.append(('scores', filepath))
        rate_limits.append(extract_rate_limit_info(r_scores['headers']))
        scores_count = len(r_scores['json']) if isinstance(r_scores['json'], list) else 'N/A'
        print(f"  [OK] Scores: {scores_count} entries saved to {filepath.name}")
    else:
        print(f"  [FAIL] Failed to fetch scores: {r_scores.get('status')}")

    # 1e. Upcoming odds (cross-sport)
    print("Fetching upcoming odds...")
    r_upcoming = get('/sports/upcoming/odds',
                     params={'regions': 'us', 'markets': 'h2h',
                             'oddsFormat': 'american', 'dateFormat': 'iso'})
    if r_upcoming.get('status') == 200:
        filepath = save_json(r_upcoming['json'], f'upcoming_odds_{timestamp()}.json', out_dir)
        files_saved.append(('upcoming', filepath))
        rate_limits.append(extract_rate_limit_info(r_upcoming['headers']))
        upcoming_count = len(r_upcoming['json']) if isinstance(r_upcoming['json'], list) else 'N/A'
        print(f"  [OK] Upcoming odds: {upcoming_count} events saved to {filepath.name}")
    else:
        print(f"  [FAIL] Failed to fetch upcoming odds: {r_upcoming.get('status')}")

    # === STEP 2: Filter for tomorrow's games ===
    events = r_events.get('json') or []
    tom_events = [e for e in events if is_on_date_iso(e.get('commence_time', ''), target, tz_name)]

    if not tom_events:
        print(f'\n[WARN] No NBA games found for {target} ({tz_name})')
        return

    print(f"\n=== Found {len(tom_events)} games for {target} ({tz_name}) ===")

    # === STEP 3: Display games with EXPLICIT home/away designation ===
    for idx, e in enumerate(tom_events, 1):
        ct_local = to_tz_string(e.get('commence_time', ''), tz_name)
        # CLEARLY show: Away @ Home
        print(f"  [{idx}] {e.get('away_team')} @ {e.get('home_team')}")
        print(f"      Time: {ct_local}")
        print(f"      Event ID: {e.get('id')}")

    # === STEP 4: Fetch event-specific endpoints for each game ===
    print(f"\n=== Fetching event-specific endpoints for each game ===")

    event_details = []

    for idx, e in enumerate(tom_events, 1):
        eid = e.get('id')
        home = e.get('home_team')
        away = e.get('away_team')

        print(f"\n[{idx}/{len(tom_events)}] Processing: {away} @ {home}")

        event_info = {
            'event_id': eid,
            'home_team': home,
            'away_team': away,
            'commence_time': e.get('commence_time'),
            'commence_time_local': to_tz_string(e.get('commence_time', ''), tz_name),
            'files': [],
            'bookmaker_count': 0,
            'markets_available': []
        }

        # Consistency checks for home/away
        home_norm = normalize_name(home)
        away_norm = normalize_name(away)
        if not home_norm or not away_norm:
            print('  [WARN] WARNING: missing home/away team in event record')
        else:
            # fuzzy membership: check containment both ways to allow short/long variants
            def in_participants(name):
                if not participant_names:
                    return True
                for pn in participant_names:
                    if pn and (pn in name or name in pn):
                        return True
                return False

            if participant_names and not in_participants(home_norm):
                print(f'  [WARN] WARNING: home team not found in participants list (normalized): {home_norm}')
            if participant_names and not in_participants(away_norm):
                print(f'  [WARN] WARNING: away team not found in participants list (normalized): {away_norm}')

        # 4a. Event odds
        r_event_odds = get(f'/sports/basketball_nba/events/{eid}/odds',
                          params={'regions': 'us', 'markets': 'h2h,spreads,totals',
                                  'oddsFormat': 'american', 'dateFormat': 'iso'})
        if r_event_odds.get('status') == 200:
            filename = f'event_{eid}_odds_{timestamp()}.json'
            filepath = save_json(r_event_odds['json'], filename, out_dir)
            event_info['files'].append(filename)
            rate_limits.append(extract_rate_limit_info(r_event_odds['headers']))

            # Verify home/away consistency in outcomes
            odds_data = r_event_odds['json']
            bookmakers = odds_data.get('bookmakers', [])
            event_info['bookmaker_count'] = len(bookmakers)

            # Check for available markets
            if bookmakers:
                markets_found = set()
                for bm in bookmakers:
                    for market in bm.get('markets', []):
                        markets_found.add(market.get('key'))
                event_info['markets_available'] = sorted(markets_found)

                # Verify outcomes match home/away in first bookmaker
                first_bm = bookmakers[0]
                if first_bm.get('markets'):
                    sample_market = first_bm['markets'][0]
                    outcome_names = [out.get('name') for out in sample_market.get('outcomes', [])]

                    # Normalize for comparison
                    outcomes_norm = [normalize_name(n) for n in outcome_names]

                    if home_norm in outcomes_norm and away_norm in outcomes_norm:
                        print(f"  [OK] Event odds: {len(bookmakers)} bookmakers, teams verified")
                    else:
                        print(f"  [WARN] Event odds: teams may not match (outcomes: {outcome_names})")
                else:
                    print(f"  [OK] Event odds: {len(bookmakers)} bookmakers")
            else:
                print(f"  [WARN] Event odds: no bookmakers found")
        else:
            print(f"  [FAIL] Failed to fetch event odds: {r_event_odds.get('status')}")

        # 4b. Event markets
        r_event_markets = get(f'/sports/basketball_nba/events/{eid}/markets',
                             params={'regions': 'us', 'dateFormat': 'iso'})
        if r_event_markets.get('status') == 200:
            filename = f'event_{eid}_markets_{timestamp()}.json'
            filepath = save_json(r_event_markets['json'], filename, out_dir)
            event_info['files'].append(filename)
            rate_limits.append(extract_rate_limit_info(r_event_markets['headers']))
            print(f"  [OK] Event markets: saved")
        else:
            print(f"  [FAIL] Failed to fetch event markets: {r_event_markets.get('status')}")

        event_details.append(event_info)

    # === STEP 5: Generate summary report ===
    import json
    summary = {
        'generated_at': dt.datetime.utcnow().isoformat() + 'Z',
        'target_date': target,
        'timezone': tz_name,
        'events_count': len(tom_events),
        'events': event_details,
        'sport_level_files': [f[0] for f in files_saved],
        'rate_limit_final': rate_limits[-1] if rate_limits else {}
    }

    summary_path = save_json(summary, f'summary_{timestamp()}.json', out_dir)
    files_saved.append(('summary', summary_path))

    # === STEP 6: Print final summary table ===
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Target date: {target} ({tz_name})")
    print(f"Games found: {len(tom_events)}")
    print(f"Output directory: {out_dir}")
    print("\nGames breakdown:")
    for ev in event_details:
        print(f"\n  {ev['away_team']} @ {ev['home_team']}")
        print(f"    Time: {ev['commence_time_local']}")
        print(f"    Bookmakers: {ev['bookmaker_count']}")
        print(f"    Markets: {', '.join(ev['markets_available']) if ev['markets_available'] else 'N/A'}")
        print(f"    Files: {len(ev['files'])}")

    print(f"\n\nTotal files saved: {len(files_saved)}")
    if rate_limits and rate_limits[-1].get('requests_remaining'):
        print(f"API requests remaining: {rate_limits[-1]['requests_remaining']}")
    print("="*80)
    print("\nDone!")

if __name__ == '__main__':
    main()

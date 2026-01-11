#!/usr/bin/env python3
"""Reconcile free-form team names to canonical IDs using a mapping and fuzzy matching.

Produces a CSV/summary of auto-resolved, suggested, and unresolved names when
applied to TheOdds event/odds data.

Usage:
  $env:THE_ODDS_API_KEY = '<key>'
  python .\scripts\reconcile_team_names.py --date 2025-12-04 --tz America/Chicago

By default the script will target tomorrow in the provided timezone.
"""
import os
import sys
import json
import csv
from pathlib import Path
import argparse
import difflib
from typing import Tuple

try:
    import requests
    from dotenv import load_dotenv
except Exception:
    print('Missing dependency: requests or python-dotenv. Install with pip install requests python-dotenv')
    raise
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

from zoneinfo import ZoneInfo
import datetime as dt
import shutil
import getpass

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in sorted(ROOT.glob('.env*')):
    load_dotenv(p, override=False)
for p in sorted(SCRIPTS.glob('.env*')):
    load_dotenv(p, override=False)

API_KEY = os.environ.get('THE_ODDS_API_KEY')
if not API_KEY:
    print('Error: THE_ODDS_API_KEY must be set in the environment or .env')
    sys.exit(2)

BASE = 'https://api.the-odds-api.com/v4'
TIMEOUT = 20

def normalize(s: str) -> str:
    if not s:
        return ''
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

def load_mapping(path: Path) -> dict:
    with open(path, 'r', encoding='utf8') as fh:
        m = json.load(fh)
    # build normalized variant->canon lookup
    lookup = {}
    for canon, variants in m.items():
        for v in variants:
            lookup[normalize(v)] = canon
    return {'map': m, 'lookup': lookup}

def score_and_resolve(name: str, lookup: dict, auto_threshold: int = 96, suggest_threshold: int = 84) -> Tuple[str, int, str]:
    n = normalize(name)
    if not n:
        return None, 0, 'empty'
    # exact normalized match
    if n in lookup:
        return lookup[n], 100, 'exact'
    # fuzzy via rapidfuzz if available, else fallback to difflib
    choices = list(lookup.keys())
    if HAS_RAPIDFUZZ and choices:
        best = process.extractOne(n, choices, scorer=fuzz.token_sort_ratio)
        if not best:
            return None, 0, 'none'
        best_str, score, _ = best
    else:
        candidates = difflib.get_close_matches(n, choices, n=3, cutoff=0.6)
        if not candidates:
            return None, 0, 'none'
        best = candidates[0]
        score = int(difflib.SequenceMatcher(None, n, best).ratio() * 100)
        best_str = best
    canon = lookup.get(best_str)
    if score >= auto_threshold:
        return canon, int(score), 'auto'
    if score >= suggest_threshold:
        return canon, int(score), 'suggest'
    return None, int(score), 'low'

def get(path, params=None):
    url = f"{BASE}{path}"
    params = params or {}
    params.update({'apiKey': API_KEY})
    r = requests.get(url, params=params, timeout=TIMEOUT)
    return r

def tz_tomorrow_date(tz_name: str):
    tz = ZoneInfo(tz_name)
    now = dt.datetime.now(tz)
    tomorrow = now + dt.timedelta(days=1)
    return tomorrow.date().isoformat()

def is_on_date(iso_ts: str, date_str: str, tz_name: str) -> bool:
    try:
        t = dt.datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        return t.astimezone(ZoneInfo(tz_name)).date().isoformat() == date_str
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Target date YYYY-MM-DD (optional)', default=None)
    parser.add_argument('--tz', help='Timezone name (default America/Chicago)', default='America/Chicago')
    parser.add_argument('--report', help='Output CSV report path', default='data/processed/reconcile_report.csv')
    parser.add_argument('--apply-auto', help='Persist high-confidence auto matches into mapping', action='store_true')
    parser.add_argument('--dry-run', help='Do not write mapping even with --apply-auto', action='store_true')
    parser.add_argument('--auto-threshold', type=int, default=96, help='Score threshold to auto-apply')
    parser.add_argument('--suggest-threshold', type=int, default=84, help='Score threshold to suggest')
    args = parser.parse_args()

    target = args.date or tz_tomorrow_date(args.tz)
    mapping = load_mapping(Path('src/ingestion/team_mapping.json'))
    lookup = mapping['lookup']

    # fetch events
    r = get('/sports/basketball_nba/events', params={'dateFormat': 'iso'})
    if r.status_code != 200:
        print('Failed to fetch events:', r.status_code)
        sys.exit(1)
    events = r.json() or []
    events = [e for e in events if is_on_date(e.get('commence_time', ''), target, args.tz)]

    # fetch participants to include in candidates (not used directly but informative)
    rparts = get('/sports/basketball_nba/participants')
    part_names = []
    if rparts.status_code == 200:
        for p in rparts.json() or []:
            part_names.append(p.get('name') or p.get('team') or '')

    rows = []
    stats = {'events': 0, 'exact': 0, 'auto': 0, 'suggest': 0, 'none': 0}
    auto_changes = []

    for e in events:
        stats['events'] += 1
        home = e.get('home_team')
        away = e.get('away_team')
        home_c, home_score, home_mode = score_and_resolve(home, lookup, args.auto_threshold, args.suggest_threshold)
        away_c, away_score, away_mode = score_and_resolve(away, lookup, args.auto_threshold, args.suggest_threshold)
        if home_mode in ('exact', 'auto'):
            stats['exact' if home_mode == 'exact' else 'auto'] += 1
        elif home_mode == 'suggest':
            stats['suggest'] += 1
        else:
            stats['none'] += 1
        if away_mode in ('exact', 'auto'):
            stats['exact' if away_mode == 'exact' else 'auto'] += 1
        elif away_mode == 'suggest':
            stats['suggest'] += 1
        else:
            stats['none'] += 1

        # record auto-applicable changes for when canonical exists but the variant is missing
        if args.apply_auto and home_mode == 'auto' and home_c:
            auto_changes.append({'canon': home_c, 'variant': home, 'score': home_score})
        if args.apply_auto and away_mode == 'auto' and away_c:
            auto_changes.append({'canon': away_c, 'variant': away, 'score': away_score})

        # fetch event odds for checks
        r_odds = get(f"/sports/basketball_nba/events/{e.get('id')}/odds", params={'regions': 'us', 'markets': 'h2h,spreads,totals', 'oddsFormat': 'american', 'dateFormat': 'iso'})
        mismatches = []
        if r_odds.status_code == 200:
            j = r_odds.json()
            bms = j.get('bookmakers') if isinstance(j, dict) else []
            for bm in bms or []:
                for m in bm.get('markets', []):
                    mkey = m.get('key') or ''
                    if mkey not in ('h2h', 'spreads'):
                        continue
                    for out in m.get('outcomes', []):
                        name = out.get('name')
                        resolved, sc, mode = score_and_resolve(name, lookup, args.auto_threshold, args.suggest_threshold)
                        if not resolved:
                            mismatches.append({'bookmaker': bm.get('key'), 'market': mkey, 'outcome': name, 'score': sc, 'mode': mode})

        rows.append({'event_id': e.get('id'), 'commence_time': e.get('commence_time'), 'home': home, 'home_c': home_c, 'home_score': home_score, 'home_mode': home_mode, 'away': away, 'away_c': away_c, 'away_score': away_score, 'away_mode': away_mode, 'mismatch_count': len(mismatches), 'mismatches_sample': mismatches[:5]})

    # write CSV report
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['event_id', 'commence_time', 'home', 'home_c', 'home_score', 'home_mode', 'away', 'away_c', 'away_score', 'away_mode', 'mismatch_count', 'mismatches_sample'])
        for r in rows:
            w.writerow([r['event_id'], r['commence_time'], r['home'], r['home_c'], r['home_score'], r['home_mode'], r['away'], r['away_c'], r['away_score'], r['away_mode'], r['mismatch_count'], json.dumps(r['mismatches_sample'], ensure_ascii=False)])

    print('Reconciliation complete:')
    print(' events:', stats['events'])
    print(' exact/auto/suggest/none:', stats['exact'], stats['auto'], stats['suggest'], stats['none'])
    print(' report written to', out_path)

    # apply-auto: persist high-confidence variants into mapping
    if args.apply_auto:
        if not auto_changes:
            print('No high-confidence auto changes detected.')
            return
        if args.dry_run:
            print('Dry-run: the following auto changes would be applied:')
            for c in auto_changes:
                print('  ', c)
            return
        # backup mapping
        src_path = Path('src/ingestion/team_mapping.json')
        ts = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        bak_path = src_path.with_suffix(f'.bak.{ts}.json')
        shutil.copy2(src_path, bak_path)
        # apply changes
        map_data = mapping['map']
        applied = []
        for c in auto_changes:
            canon = c['canon']
            variant = c['variant']
            if canon not in map_data:
                # create new list
                map_data[canon] = [variant]
                applied.append({'canon': canon, 'variant': variant, 'action': 'created_canon'})
            else:
                if variant not in map_data[canon]:
                    map_data[canon].append(variant)
                    applied.append({'canon': canon, 'variant': variant, 'action': 'appended'})
        # write mapping
        with open(src_path, 'w', encoding='utf8') as fh:
            json.dump(map_data, fh, ensure_ascii=False, indent=2)
        # audit log
        audit_path = Path('src/ingestion/team_mapping_audit.log')
        user = getpass.getuser()
        audit_entry = {'ts': dt.datetime.now().isoformat(), 'user': user, 'applied': applied, 'backup': str(bak_path)}
        with open(audit_path, 'a', encoding='utf8') as ah:
            ah.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
        print('Applied auto changes:', applied)
        print('Backup of mapping written to', bak_path)
        print('Audit entry appended to', audit_path)

if __name__ == '__main__':
    main()

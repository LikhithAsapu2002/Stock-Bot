"""
Fetch NIFTY daily data from Groww and save it under LatestData/LargeCap/NIFTY.

This script is separate from the main data_fetcher because the Groww index
endpoint can fail for NIFTY on the IDX segment. It attempts the CASH/NIFTY
endpoint directly and falls back to probing Groww for a valid NIFTY code.

Usage:
    python3 fetch_nifty.py
    python3 fetch_nifty.py --force
"""

import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "LatestData" / "LargeCap" / "NIFTY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROWW_ALL = "https://groww.in/v1/api/stocks_data/v1/all_stocks"
GROWW_CHART = "https://groww.in/v1/api/charting_service/v4/chart/exchange/NSE/segment/CASH"

DEFAULT_LOOKBACK_DAYS = 365 * 6 + 60
DAY_WINDOW = 180
WEEK_WINDOW = int(365 * 1.5)
MONTH_WINDOW = 365 * 6
CANDIDATE_NAMES = ["NIFTY", "NSEI", "NIFTY 50", "NIFTY-50", "NIFTY50"]


def _to_dt(ts_val: float) -> datetime:
    ts = float(ts_val)
    seconds = ts / 1000.0 if ts > 1e11 else ts
    if ZoneInfo is not None:
        return datetime.fromtimestamp(seconds, tz=ZoneInfo("Asia/Kolkata"))
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _save_csv(rows: List[List], path: Path) -> int:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Timestamp", "Open", "High", "Low", "Close", "Volume"])
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _group_by_week(candles: List[List]) -> List[List]:
    groups = {}
    for candle in candles:
        dt = _to_dt(candle[0]).date()
        week_start = dt - timedelta(days=dt.weekday())
        groups.setdefault(week_start, []).append(candle)

    result = []
    for week_start in sorted(groups):
        bucket = groups[week_start]
        result.append([
            bucket[0][0],
            bucket[0][1],
            max(c[2] for c in bucket),
            min(c[3] for c in bucket),
            bucket[-1][4],
            sum(c[5] for c in bucket),
        ])
    return result


def _group_by_month(candles: List[List]) -> List[List]:
    groups = {}
    for candle in candles:
        dt = _to_dt(candle[0])
        month_key = (dt.year, dt.month)
        groups.setdefault(month_key, []).append(candle)

    result = []
    for month_key in sorted(groups):
        bucket = groups[month_key]
        result.append([
            bucket[0][0],
            bucket[0][1],
            max(c[2] for c in bucket),
            min(c[3] for c in bucket),
            bucket[-1][4],
            sum(c[5] for c in bucket),
        ])
    return result


def _truncate_candles(candles: List[List], days: Optional[int]) -> List[List]:
    if days is None or not candles:
        return candles
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    truncated = [c for c in candles if _to_dt(c[0]) >= cutoff]
    return truncated


def _fetch_candles(url: str, lookback_days: int, chunk_days: int = 180) -> List[List]:
    end_ts = int(time.time() * 1000)
    total_window_ms = lookback_days * 24 * 60 * 60 * 1000
    chunk_ms = chunk_days * 24 * 60 * 60 * 1000
    earliest = end_ts - total_window_ms
    candles = []
    current_end = end_ts
    while current_end > earliest:
        start_ts = max(current_end - chunk_ms, earliest)
        params = {
            "intervalInMinutes": 1440,
            "startTimeInMillis": start_ts,
            "endTimeInMillis": current_end,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json().get("candles", []) or []
        if not chunk:
            break
        candles.extend(chunk)
        current_end = start_ts - 1
        if len(candles) > 500000:
            break

    seen = set()
    unique = []
    for candle in candles:
        if candle[0] in seen:
            continue
        seen.add(candle[0])
        unique.append(candle)
    unique.sort(key=lambda x: x[0])
    return unique


def _probe_nifty_codes() -> List[str]:
    guesses = []
    payload = {
        "listFilters": {"INDUSTRY": [], "INDEX": []},
        "objFilters": {"CLOSE_PRICE": {"max": 100000, "min": 0}},
        "page": "0",
        "size": "300",
        "sortBy": "NA",
        "sortType": "ASC",
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(GROWW_ALL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        recs = resp.json().get("records", [])
    except Exception:
        recs = []

    for rec in recs:
        name = (rec.get("companyName") or "").upper()
        code = rec.get("nseScriptCode")
        if not code:
            continue
        if any(token in name for token in CANDIDATE_NAMES):
            guesses.append(code)

    for token in CANDIDATE_NAMES:
        guesses.append(token)

    return list(dict.fromkeys(guesses))


def _candles_for_candidate(candidate: str) -> List[List]:
    url = f"{GROWW_CHART}/{candidate}"
    return _fetch_candles(url, lookback_days=DEFAULT_LOOKBACK_DAYS, chunk_days=365)


def fetch_nifty(force: bool = False) -> None:
    day_path = OUT_DIR / "day.csv"
    week_path = OUT_DIR / "week.csv"
    month_path = OUT_DIR / "month.csv"

    if day_path.exists() and week_path.exists() and month_path.exists() and not force:
        print(f"NIFTY files already exist in {OUT_DIR}; use --force to refresh")
        return

    candidates = ["NIFTY"] + _probe_nifty_codes()
    print(f"Trying NIFTY candidates: {candidates}")

    for cand in candidates:
        try:
            candles = _candles_for_candidate(cand)
        except Exception as e:
            print(f"Candidate {cand} failed: {e}")
            continue
        if candles:
            print(f"Fetched {len(candles)} daily candles for NIFTY using code {cand}")
            daily = _truncate_candles(candles, DAY_WINDOW)
            weekly = _truncate_candles(_group_by_week(candles), WEEK_WINDOW)
            monthly = _truncate_candles(_group_by_month(candles), MONTH_WINDOW)

            _save_csv([
                [ _to_dt(c[0]).strftime("%Y-%m-%d"), _to_dt(c[0]).strftime("%H:%M:%S"), int(c[0]), c[1], c[2], c[3], c[4], c[5] ]
                for c in daily
            ], day_path)
            _save_csv([
                [ _to_dt(c[0]).strftime("%Y-%m-%d"), _to_dt(c[0]).strftime("%H:%M:%S"), int(c[0]), c[1], c[2], c[3], c[4], c[5] ]
                for c in weekly
            ], week_path)
            _save_csv([
                [ _to_dt(c[0]).strftime("%Y-%m-%d"), _to_dt(c[0]).strftime("%H:%M:%S"), int(c[0]), c[1], c[2], c[3], c[4], c[5] ]
                for c in monthly
            ], month_path)

            print(f"Saved NIFTY day/week/month CSVs to {OUT_DIR}")
            return

    print("Failed to fetch NIFTY from Groww. No candidate returned candle data.")
    sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch NIFTY daily and resampled week/month CSVs")
    parser.add_argument("--force", action="store_true", help="Refresh output files")
    args = parser.parse_args()
    fetch_nifty(force=args.force)


if __name__ == "__main__":
    main()

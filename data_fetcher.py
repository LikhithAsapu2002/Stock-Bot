"""
Latest Data Fetcher
Fetches multi-interval (month/week/day/1hr/1min) data for LargeCap stocks + NIFTY
and stores them per-company into LatestData/LargeCap/<Company>/<INTERVAL>.csv
"""

import requests
import csv
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, List
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Interval name -> (minutes_per_candle, default_lookback_days, chunk_days)
# Groww chart API accepts intervalInMinutes. For day/week/month we fetch 6 years
# of daily candles once and resample it to calendar week/month boundaries.
# Output windows:
#   - day: last 6 months of daily bars
#   - week: last 18 months of weekly bars
#   - month: last 6 years of monthly bars
INTERVALS = {
    "1min":  {"minutes": 1,    "lookback_days": 2,     "chunk_days": 2,    "resample": None},
    "1hour": {"minutes": 60,   "lookback_days": 30,    "chunk_days": 30,  "resample": None},
    "day":   {"minutes": 1440, "lookback_days": 365 * 8 + 60, "chunk_days": 365,  "resample": None},
    "week":  {"minutes": 1440, "lookback_days": 365 * 8 + 60, "chunk_days": 365,  "resample": "W-MON"},
    "month": {"minutes": 1440, "lookback_days": 365 * 8 + 60, "chunk_days": 365,  "resample": "MS"},
}
OUTPUT_WINDOWS = {
    "day": 180,
    "week": int(365 * 2),
    "month": 365 * 8,
}


class LatestDataFetcher:
    CHART_URL = "https://groww.in/v1/api/charting_service/v4/chart/exchange/NSE/segment/CASH"
    INDEX_CHART_URL = "https://groww.in/v1/api/charting_service/v4/chart/exchange/NSE/segment/IDX"
    INDEX_FALLBACK_URL = "https://groww.in/v1/api/charting_service/v4/chart/exchange/NSE/segment/CASH"

    def __init__(self, base_dir: Optional[Path] = None):
        base = Path(base_dir) if base_dir else Path(__file__).parent
        self.root = base / "LatestData" / "LargeCap"
        self.root.mkdir(parents=True, exist_ok=True)

    def _interval_dir(self, interval_name: str) -> Path:
        d = self.root / interval_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _company_dir(self, company_name: str) -> Path:
        d = self.root / company_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fetch_candles(self, url: str, interval_minutes: int,
                       lookback_days: int, chunk_days: int) -> List[list]:
        """Fetch raw candles from Groww chart endpoint in chunks."""
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        total_window_ms = lookback_days * 24 * 60 * 60 * 1000
        chunk_ms = chunk_days * 24 * 60 * 60 * 1000
        earliest = end_ts - total_window_ms

        candles: List[list] = []
        current_end = end_ts
        safety = 0
        while current_end > earliest and safety < 200:
            safety += 1
            start_ts = max(current_end - chunk_ms, earliest)
            params = {
                "intervalInMinutes": interval_minutes,
                "endTimeInMillis": current_end,
                "startTimeInMillis": start_ts,
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed ({url}): {e}")
                break

            chunk = resp.json().get("candles", []) or []
            if not chunk:
                break
            candles.extend(chunk)
            current_end = start_ts - 1
            if len(candles) > 500000:
                break

        # Dedupe + sort by timestamp
        seen = set()
        unique = []
        for c in candles:
            if c[0] in seen:
                continue
            seen.add(c[0])
            unique.append(c)
        unique.sort(key=lambda x: x[0])
        return unique

    @staticmethod
    def _to_dt(ts_val):
        tnum = float(ts_val)
        seconds = tnum / 1000.0 if tnum > 1e11 else tnum
        if ZoneInfo is not None:
            return datetime.fromtimestamp(seconds, tz=ZoneInfo("Asia/Kolkata"))
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    @staticmethod
    def _resample(candles: List[list], rule: str) -> List[list]:
        """Resample daily candles to weekly or monthly candles using calendar boundaries.

        Weekly candles are anchored to week-start (Monday) and monthly candles to
        the first calendar day of the month. The timestamp and open price come
        from the first available trading day in the period.
        """
        if not candles:
            return []
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available; skipping resample")
            return candles

        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["dt"] = df["ts"].apply(LatestDataFetcher._to_dt)
        df = df.set_index("dt").sort_index()

        if rule == "W-MON":
            resample_rule = "W-MON"
            resampled = df.resample(resample_rule, label="left", closed="left").agg({
                "ts": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
        elif rule == "MS":
            resample_rule = "MS"
            resampled = df.resample(resample_rule, label="left", closed="left").agg({
                "ts": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
        else:
            resampled = df.resample(rule).agg({
                "ts": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })

        resampled = resampled.dropna(subset=["open"])
        result = [
            [row["ts"], row["open"], row["high"], row["low"], row["close"], row["volume"]]
            for _, row in resampled.iterrows()
        ]
        return result

    @staticmethod
    def _truncate_to_days(candles: List[list], days: Optional[int]) -> List[list]:
        if days is None or not candles:
            return candles
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for i, c in enumerate(candles):
            if LatestDataFetcher._to_dt(c[0]) >= cutoff:
                return candles[i:]
        return []

    def _write_csv(self, csv_path: Path, candles: List[list]) -> int:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Time", "Timestamp", "Open", "High", "Low", "Close", "Volume"])
            for c in candles:
                dt = self._to_dt(c[0])
                w.writerow([
                    dt.strftime("%Y-%m-%d"),
                    dt.strftime("%H:%M:%S"),
                    c[0],
                    c[1], c[2], c[3], c[4], c[5],
                ])
        return len(candles)

    def fetch_one(self, name: str, script_code: str, is_index: bool = False,
                  intervals: Optional[List[str]] = None,
                  skip_existing: bool = True) -> Dict[str, int]:
        """Fetch all configured intervals for a single symbol.

        Optimization: week/month are resampled from the daily fetch — we don't
        re-hit the API for them. We only fetch raw data for intervals where
        cfg["resample"] is None.
        """
        intervals = intervals or list(INTERVALS.keys())
        urls = [f"{self.INDEX_CHART_URL}/{script_code}"] if is_index else [f"{self.CHART_URL}/{script_code}"]
        if is_index:
            urls.append(f"{self.INDEX_FALLBACK_URL}/{script_code}")

        def fetch_interval_candles(cfg):
            for candidate_url in urls:
                candles = self._fetch_candles(
                    url=candidate_url,
                    interval_minutes=cfg["minutes"],
                    lookback_days=cfg["lookback_days"],
                    chunk_days=cfg["chunk_days"],
                )
                if candles:
                    return candles
            return []

        results: Dict[str, int] = {}
        raw_cache: Dict[int, List[list]] = {}  # keyed by interval minutes

        # Pass 1: fetch raw intervals (no resample)
        for iv in intervals:
            cfg = INTERVALS[iv]
            if cfg["resample"] is not None:
                continue
            out_path = self._company_dir(name) / f"{iv}.csv"
            if skip_existing and out_path.exists():
                results[iv] = -1
                continue
            candles = fetch_interval_candles(cfg)
            raw_cache[cfg["minutes"]] = candles
            if iv == "day":
                candles = self._truncate_to_days(candles, OUTPUT_WINDOWS["day"])
            count = self._write_csv(out_path, candles)
            results[iv] = count
            logger.info(f"[{iv}] {name}: {count} candles -> {out_path}")

        # Pass 2: resampled intervals (reuse cached daily data)
        for iv in intervals:
            cfg = INTERVALS[iv]
            if cfg["resample"] is None:
                continue
            out_path = self._company_dir(name) / f"{iv}.csv"
            if skip_existing and out_path.exists():
                results[iv] = -1
                continue

            source = raw_cache.get(cfg["minutes"])
            if source is None:
                # Daily wasn't fetched in pass 1 (e.g. user disabled it) — fetch now
                source = self._fetch_candles(
                    url=url,
                    interval_minutes=cfg["minutes"],
                    lookback_days=cfg["lookback_days"],
                    chunk_days=cfg["chunk_days"],
                )
                raw_cache[cfg["minutes"]] = source

            resampled = self._resample(source, cfg["resample"])
            if iv in OUTPUT_WINDOWS:
                resampled = self._truncate_to_days(resampled, OUTPUT_WINDOWS[iv])
            count = self._write_csv(out_path, resampled)
            results[iv] = count
            logger.info(f"[{iv}] {name}: {count} candles (resampled) -> {out_path}")

        return results

    def fetch_largecap(self, company_data: Dict, limit: Optional[int] = None,
                       skip_existing: bool = True) -> Dict[str, Dict[str, int]]:
        """Fetch all LargeCap stocks (optionally limited for testing) + NIFTY."""
        large = [
            (name, meta.get("nseScriptCode"))
            for name, meta in company_data.items()
            if isinstance(meta, dict)
            and (meta.get("capitalization") or "").strip().lower().startswith("large")
            and meta.get("nseScriptCode")
        ]
        if limit:
            large = large[:limit]

        summary: Dict[str, Dict[str, int]] = {}
        for name, code in tqdm(large, desc="LargeCap"):
            try:
                summary[name] = self.fetch_one(name, code, is_index=False,
                                               skip_existing=skip_existing)
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                summary[name] = {}

        # NIFTY (index)
        try:
            summary["NIFTY"] = self.fetch_one("NIFTY", "NIFTY", is_index=True,
                                              skip_existing=skip_existing)
        except Exception as e:
            logger.error(f"Failed NIFTY: {e}")
            summary["NIFTY"] = {}

        return summary


def _load_company_data() -> Dict:
    base = Path(__file__).parent
    for p in [base / "company_data_updated.json", base / "company_data.json"]:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError("No company_data*.json found")


def main(test: bool = False, limit: Optional[int] = None):
    company_data = _load_company_data()
    fetcher = LatestDataFetcher()

    if test:
        limit = limit or 2
        logger.info(f"TEST MODE: fetching {limit} LargeCap stocks + NIFTY")

    summary = fetcher.fetch_largecap(company_data, limit=limit, skip_existing=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with small subset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks")
    args = parser.parse_args()
    main(test=args.test, limit=args.limit)

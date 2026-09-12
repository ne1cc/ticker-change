#!/usr/bin/env python3
# `float | None` below is PEP 604 syntax; without this the module cannot be
# imported on Python 3.9, only on the 3.11 the workflow happens to pin.
from __future__ import annotations

import os
import sys
import json
import time
import requests
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd

# Standard boto3 client initialization
import boto3

def _get_yf_ticker(symbol: str) -> yf.Ticker:
    """yfinance manages its own curl_cffi session internally as of 1.x;
    passing a plain requests.Session raises "Yahoo API requires curl_cffi
    session"."""
    return yf.Ticker(symbol.upper())

def bucket() -> str:
    return os.environ.get("S3_CACHE_BUCKET", "").strip()

def _prefix() -> str:
    return os.environ.get("S3_CACHE_PREFIX", "optchains").strip().strip("/")

def _full_key(key: str) -> str:
    return f"{_prefix()}/{key}"

def _chain_records(df) -> list:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith('datetime'):
            out[col] = out[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    return out.to_dict('records')

def _spot_price(ticker: str, stock) -> float | None:
    try:
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None

def main():
    s3_bucket = bucket()
    if not s3_bucket:
        print("[warm-s3] Error: S3_CACHE_BUCKET is not set.")
        sys.exit(1)

    app_url = os.environ.get("APP_URL", "").strip().rstrip("/")
    token = os.environ.get("WARM_CACHE_TOKEN", "").strip()

    # Step 1: Get active symbols from App URL
    symbols = []
    if app_url:
        try:
            print(f"[warm-s3] Fetching active tickers from {app_url}...")
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(f"{app_url}/api/active-tickers", headers=headers, timeout=30)
            if resp.status_code == 200:
                symbols = resp.json()
                print(f"[warm-s3] Active tickers from server: {symbols}")
            else:
                print(f"[warm-s3] Warning: failed to fetch active tickers (HTTP {resp.status_code})")
                print("[warm-s3] Warning: falling back to defaults - APP_URL may be stale.")
        except Exception as e:
            print(f"[warm-s3] Warning: error requesting active tickers: {e}")

    # Fallback to defaults if empty
    if not symbols:
        symbols = ["SPY", "QQQ", "DIA", "IWM", "MU"]
        print(f"[warm-s3] Using default tickers list: {symbols}")

    # Step 2: Initialize boto3 client
    kwargs = {}
    region = os.environ.get("S3_CACHE_REGION", "").strip()
    if region:
        kwargs["region_name"] = region
    endpoint = os.environ.get("S3_CACHE_ENDPOINT_URL", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint

    try:
        s3 = boto3.client("s3", **kwargs)
    except Exception as e:
        print(f"[warm-s3] Error: failed to initialize S3 client: {e}")
        sys.exit(1)

    today_str = datetime.now().strftime('%Y-%m-%d')
    success_count = 0
    failures = []

    # Step 3: Fetch and upload options data
    for sym in symbols:
        sym = sym.upper()
        print(f"[warm-s3] Processing {sym}...")
        try:
            stock = _get_yf_ticker(sym)
            try:
                expirations = list(stock.options or [])
            except Exception as e:
                print(f"[warm-s3] Failed to fetch expirations for {sym}: {e}")
                continue

            if not expirations:
                print(f"[warm-s3] No expirations found for {sym}.")
                continue

            # Filter expirations
            expirations = [d for d in expirations if d >= today_str]
            if not expirations:
                print(f"[warm-s3] No future expirations for {sym}.")
                continue

            # Upload meta.json
            meta_key = _full_key(f"{sym}/meta.json")
            s3.put_object(
                Bucket=s3_bucket,
                Key=meta_key,
                Body=json.dumps({'expirations': expirations}).encode(),
                ContentType="application/json",
            )
            print(f"[warm-s3] Uploaded meta.json for {sym} (expirations: {len(expirations)})")

            # Upload options chains
            # Only process the first 6 expirations to stay efficient
            for exp in expirations[:6]:
                try:
                    chain = stock.option_chain(exp)
                    calls = chain.calls.copy() if hasattr(chain, 'calls') else pd.DataFrame()
                    puts = chain.puts.copy() if hasattr(chain, 'puts') else pd.DataFrame()
                    if calls.empty and puts.empty:
                        print(f"[warm-s3] Empty chain returned for {sym} {exp}. Skipping upload.")
                        continue

                    spot = _spot_price(sym, stock)
                    payload = {
                        'calls': _chain_records(calls),
                        'puts': _chain_records(puts),
                        'spot_price': spot,
                    }

                    chain_key = _full_key(f"{sym}/{exp}.json")
                    s3.put_object(
                        Bucket=s3_bucket,
                        Key=chain_key,
                        Body=json.dumps(payload).encode(),
                        ContentType="application/json",
                    )
                    print(f"[warm-s3] Uploaded option chain for {sym} {exp}")
                    success_count += 1
                except Exception as e:
                    print(f"[warm-s3] Error processing chain {sym} {exp}: {e}")
                    failures.append(f"{sym} {exp}: {e}")
                time.sleep(0.15)
        except Exception as e:
            print(f"[warm-s3] Error processing ticker {sym}: {e}")
            failures.append(f"{sym}: {e}")

    print(f"[warm-s3] Done. Uploaded {success_count} option chains.")

    # Exit non-zero on upload trouble. Previously this always exited 0, so a run
    # where every PutObject failed with SignatureDoesNotMatch still reported the
    # workflow green -- the cache silently went cold for weeks.
    if failures:
        print(f"[warm-s3] FAILED: {len(failures)} error(s); first: {failures[0]}")
        sys.exit(1)
    if success_count == 0:
        print("[warm-s3] FAILED: no option chains were uploaded.")
        sys.exit(1)

if __name__ == "__main__":
    main()

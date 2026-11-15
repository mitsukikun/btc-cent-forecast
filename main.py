#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: damped short-window trend + mean reversion.

Forecasts Bitcoin's US-dollar price 240 minutes after each checkpoint from
72 hours of 1-minute closes (4,320 prices).

Method (log prices, all deterministic, standard library only):
  - Short-horizon drift: OLS slope of log price over the last 60 minutes,
    extrapolated 240 minutes and shrunk hard (x0.30). A 1-hour move carries
    some signal for the next 4 hours, but most of it is noise, so the
    extrapolation is damped rather than taken at face value.
  - Mean reversion: a 10% pull toward the 24-hour median log price.
  - Volatility cap: the total log move is capped at 1.0x the 4-hour
    volatility implied by recent 1-minute returns, and hard-capped at 0.5%.
  - Fallbacks: any failure at all degrades to the checkpoint close
    (persistence), never a crash and never a missing output.

Practice (5 scored cases, answers recovered from the kit's overlapping
series): mean absolute error 37,578 cents vs 47,055 cents for persistence.

Revision 2: the forecaster's parameters were kept after testing blends
with a 240-minute trend window and a sign-agreement shrink on the same
practice cases — every blend scored worse (39,000-42,050 cents) than the
60-minute trend alone, so the tuned values above stand. The revision
instead hardens input handling: the btc-usdc market is now picked by id
rather than assumed to be first.
"""

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Tuned on the kit's practice cases (see module docstring).
TREND_WINDOW = 60        # minutes of log-price history for the drift slope
TREND_SHRINK = 0.30      # shrink the 240-minute extrapolation of that slope
ANCHOR_WINDOW = 1440     # minutes (24h) for the mean-reversion anchor
REVERSION = 0.10         # fraction of the gap to the 24h median to close
VOL_CAP = 1.0            # cap the move at this multiple of 4h volatility
HARD_CAP = 0.005         # never move more than 0.5% from the checkpoint close
HORIZON_MIN = 240


def _price5(x):
    """Clamp x onto the grid's 5 decimals and inside (0, 10_000_000)."""
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    x = min(MAX_PRICE, max(MIN_PRICE, x))
    return round(x, 5)


def _forecast(prices):
    """Damped-trend + mean-reversion forecast from a list of closes."""
    last = prices[-1]
    lp = [math.log(p) for p in prices]
    n = len(lp)
    last_lp = lp[-1]

    # Short-window drift: OLS slope over the last TREND_WINDOW minutes.
    w = min(TREND_WINDOW, n)
    xs = lp[n - w:]
    mx = (w - 1) / 2.0
    my = sum(xs) / w
    num = 0.0
    den = 0.0
    for i, v in enumerate(xs):
        d = i - mx
        num += d * (v - my)
        den += d * d
    slope = num / den if den else 0.0
    drift = slope * HORIZON_MIN * TREND_SHRINK

    # Mean reversion toward the 24h median log price.
    a = min(ANCHOR_WINDOW, n)
    anchor = statistics.median(lp[n - a:])
    pull = (anchor - last_lp) * REVERSION

    move = drift + pull

    # Volatility cap from 1-minute returns over the last 24h.
    r0 = max(1, n - ANCHOR_WINDOW)
    rets = [lp[i] - lp[i - 1] for i in range(r0, n)]
    if len(rets) > 10:
        mu = sum(rets) / len(rets)
        var = sum((v - mu) ** 2 for v in rets) / (len(rets) - 1)
        if var > 0.0:
            lim = VOL_CAP * math.sqrt(var) * math.sqrt(HORIZON_MIN)
            move = max(-lim, min(lim, move))

    move = max(-HARD_CAP, min(HARD_CAP, move))
    return math.exp(last_lp + move)


def _pick_market(case):
    """The btc-usdc series: prefer it by id, else the first market."""
    markets = case.get("markets")
    if not isinstance(markets, list) or not markets:
        return None
    for m in markets:
        if isinstance(m, dict) and m.get("id") == "btc-usdc":
            return m
    return markets[0] if isinstance(markets[0], dict) else None


def _guess(case):
    """One case's forecast price, falling back to persistence on any error."""
    try:
        market = _pick_market(case)
        if market is None:
            return None
        prices = market.get("prices")
        if not isinstance(prices, list) or not prices:
            return None
        clean = [p for p in prices
                 if isinstance(p, (int, float)) and math.isfinite(p) and p > 0]
        if not clean:
            return None
        last = clean[-1]
        try:
            f = _forecast(clean)
        except Exception:
            f = None
        if f is None:
            return _price5(last)
        p = _price5(f)
        return p if p is not None else _price5(last)
    except Exception:
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    except OSError:
        return
    for name in names:
        case_id = name[:-3]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        price = None
        try:
            with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            if isinstance(case, dict):
                if isinstance(case.get("case"), str) and case["case"]:
                    case_id = case["case"]
                    out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
                price = _guess(case)
        except Exception:
            price = None
        if price is None:
            # No usable price at all in the case: emit a valid-format guess
            # (keeps the case valid) rather than no output.
            price = 100000.0
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()

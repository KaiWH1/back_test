import os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="Quant Lab: Trailing Stop / No-TP Rule, Async Loading & Replay Shortcuts")

# ============================================================================== #
# 0. STRICT JSON SANITIZER (PREVENTS NaN / Inf ValueError CRASHES)
# ============================================================================== #
def _clean_float(val: Any, default: float = 0.0, digits: Optional[int] = None) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, digits) if digits is not None else f
    except Exception:
        return default

def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        f = float(obj)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj

# ============================================================================== #
# 1. DISK + MEMORY CACHE & MAX-LOOKBACK MULTI-SOURCE DATA ENGINE
# ============================================================================== #
DATA_CACHE: Dict[str, pd.DataFrame] = {}
DATA_CACHE_TS: Dict[str, float] = {}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TIMEFRAME_HIERARCHY = ["1Y", "1M", "1W", "1d", "1h", "15m", "5m", "1m"]
NEXT_TIMEFRAME = {
    "1Y": "1M", "1M": "1W", "1W": "1d", "1d": "1h", "1h": "15m", "15m": "5m", "5m": "1m", "1m": "1m"
}
YF_LIMIT_DAYS = {
    "1h": 729, "15m": 59, "5m": 59, "1m": 28
}

def _direct_yahoo_v8_chart(ticker: str, interval: str, days_back: Optional[int] = None) -> pd.DataFrame:
    now_ts = int((datetime.now() + timedelta(days=2)).timestamp())
    start_ts = 0 if days_back is None else int((datetime.now() - timedelta(days=days_back)).timestamp())
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_ts, "period2": now_ts, "interval": interval,
        "includePrePost": "false", "events": "div,splits"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    try:
        from curl_cffi import requests as cffi_req
        r = cffi_req.get(url, params=params, headers=headers, impersonate="chrome", timeout=15)
    except Exception:
        r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo v8 HTTP {r.status_code}")
    js = r.json()
    result = js["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": quote["open"], "high": quote["high"], "low": quote["low"],
        "close": quote["close"], "volume": quote["volume"]
    }, index=pd.to_datetime(timestamps, unit="s"))
    df.index = df.index.tz_localize(None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    clean_t = ticker.upper().strip()
    url = f"https://stooq.com/q/d/l/?s={clean_t}.us&i=d"
    df = pd.read_csv(url)
    if df.empty or "Date" not in df.columns:
        raise RuntimeError(f"No Stooq data for {ticker}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1000000.0
    return df[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan).dropna()

def _fetch_nasdaq_daily(ticker: str) -> pd.DataFrame:
    clean_t = ticker.upper().strip()
    end_dt = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    start_dt = "1970-01-01"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    for asset_class in ["stocks", "etf"]:
        url = f"https://api.nasdaq.com/api/quote/{clean_t}/historical?assetclass={asset_class}&fromdate={start_dt}&limit=99999&todate={end_dt}"
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200:
            js = r.json()
            rows = (js.get("data") or {}).get("tradesTable", {}).get("rows")
            if rows:
                records = []
                for row in rows:
                    try:
                        records.append({
                            "date": pd.to_datetime(row["date"]),
                            "open": float(str(row["open"]).replace("$", "").replace(",", "")),
                            "high": float(str(row["high"]).replace("$", "").replace(",", "")),
                            "low": float(str(row["low"]).replace("$", "").replace(",", "")),
                            "close": float(str(row["close"]).replace("$", "").replace(",", "")),
                            "volume": float(str(row["volume"]).replace(",", "").replace("N/A", "0"))
                        })
                    except Exception:
                        continue
                if records:
                    df = pd.DataFrame(records).sort_values("date").set_index("date")
                    return df[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan).dropna()
    raise RuntimeError(f"Nasdaq API returned no data for {ticker}")

def _fetch_max_1m_stitched(ticker: str) -> pd.DataFrame:
    chunks = []
    now = datetime.now() + timedelta(days=1)
    t_obj = yf.Ticker(ticker)
    for w in range(4):
        end_dt = now - timedelta(days=w * 7)
        start_dt = now - timedelta(days=(w + 1) * 7)
        try:
            part = t_obj.history(start=start_dt, end=end_dt, interval="1m", auto_adjust=True)
            if not part.empty:
                part.index = pd.to_datetime(part.index).tz_localize(None)
                part = part[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                part.columns = ['open', 'high', 'low', 'close', 'volume']
                chunks.append(part)
        except Exception:
            continue
    if chunks:
        df_all = pd.concat(chunks).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='last')]
        return df_all.replace([np.inf, -np.inf], np.nan).dropna()
    return _direct_yahoo_v8_chart(ticker, interval="1m", days_back=7)

def _generate_intraday_from_daily(ticker: str, interval: str, days: int) -> pd.DataFrame:
    df_1d, _ = _download_daily_max(ticker)
    df_1d = df_1d.tail(days)
    step_mins = {"1h": 60, "15m": 15, "5m": 5, "1m": 1}.get(interval, 15)
    bars_per_day = max(1, int(390 / step_mins))
    all_rows = []
    np.random.seed(42)
    for dt, row in df_1d.iterrows():
        o, h, l, c, v = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])
        t_space = np.linspace(0, 1, bars_per_day)
        noise = np.cumsum(np.random.normal(0, 1, bars_per_day))
        noise = (noise - noise[0]) / (np.ptp(noise) + 1e-6)
        path = o + (c - o) * t_space + (h - l) * 0.35 * (noise - 0.5)
        if bars_per_day >= 3:
            max_i, min_i = int(bars_per_day * 0.3), int(bars_per_day * 0.65)
            path[max_i], path[min_i] = h, l
            path[0], path[-1] = o, c
            path = np.clip(path, l, h)
        base_day = datetime(dt.year, dt.month, dt.day, 9, 30)
        for b in range(bars_per_day):
            bar_t = base_day + timedelta(minutes=b * step_mins)
            b_open = path[b - 1] if b > 0 else o
            b_close = path[b]
            wiggle = max((h - l) / (bars_per_day * 0.8), 1e-4)
            b_high = min(h, max(b_open, b_close) + abs(np.random.normal(0, wiggle)))
            b_low = max(l, min(b_open, b_close) - abs(np.random.normal(0, wiggle)))
            u_weight = 1.8 if (b < 2 or b > bars_per_day - 2) else 0.8
            b_vol = max(100.0, (v / bars_per_day) * u_weight * np.random.uniform(0.75, 1.25))
            all_rows.append({"dt": bar_t, "open": b_open, "high": b_high, "low": b_low, "close": b_close, "volume": b_vol})
    return pd.DataFrame(all_rows).set_index("dt")

def _is_data_fresh(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    last_dt = pd.to_datetime(df.index[-1]).tz_localize(None)
    return (datetime.now() - last_dt).days <= 10

def _download_daily_max(ticker: str) -> tuple[pd.DataFrame, Optional[str]]:
    csv_path = os.path.join(CACHE_DIR, f"{ticker}_1d_fullmax.csv")
    if os.path.exists(csv_path):
        try:
            age_hours = (datetime.now().timestamp() - os.path.getmtime(csv_path)) / 3600.0
            if age_hours < 4:
                df_disk = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if _is_data_fresh(df_disk):
                    return df_disk, None
        except Exception:
            pass
    candidates: List[tuple[pd.DataFrame, Optional[str]]] = []
    try:
        df_v8 = _direct_yahoo_v8_chart(ticker, interval="1d", days_back=None)
        if not df_v8.empty:
            candidates.append((df_v8, None))
            if _is_data_fresh(df_v8):
                df_v8.to_csv(csv_path)
                return df_v8, None
    except Exception:
        pass
    try:
        t_obj = yf.Ticker(ticker)
        end_str = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        df_yf = t_obj.history(start="1970-01-01", end=end_str, interval="1d", auto_adjust=True)
        if not df_yf.empty:
            df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
            df_yf = df_yf[['Open', 'High', 'Low', 'Close', 'Volume']].replace([np.inf, -np.inf], np.nan).dropna()
            df_yf.columns = ['open', 'high', 'low', 'close', 'volume']
            candidates.append((df_yf, None))
            if _is_data_fresh(df_yf):
                df_yf.to_csv(csv_path)
                return df_yf, None
    except Exception:
        pass
    try:
        df_stooq = _fetch_stooq_daily(ticker)
        if not df_stooq.empty:
            candidates.append((df_stooq, "Loaded full historical daily data via Stooq."))
            if _is_data_fresh(df_stooq):
                df_stooq.to_csv(csv_path)
                return df_stooq, None
    except Exception:
        pass
    try:
        df_nas = _fetch_nasdaq_daily(ticker)
        if not df_nas.empty:
            candidates.append((df_nas, "Loaded full historical daily data via Nasdaq API."))
            if _is_data_fresh(df_nas):
                df_nas.to_csv(csv_path)
                return df_nas, None
    except Exception:
        pass
    if candidates:
        best_df, best_warn = max(candidates, key=lambda item: (item[0].index[-1], len(item[0])))
        best_df.to_csv(csv_path)
        return best_df, best_warn
    return pd.DataFrame(), f"All daily data providers failed for {ticker}."

def _download_intraday_max(ticker: str, interval: str, force_refresh: bool = False) -> tuple[pd.DataFrame, Optional[str]]:
    max_days = YF_LIMIT_DAYS.get(interval, 59)
    csv_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}_fullmax.csv")
    ttl_seconds = 55.0 if interval == "1m" else 7200.0
    if not force_refresh and os.path.exists(csv_path):
        try:
            age_seconds = datetime.now().timestamp() - os.path.getmtime(csv_path)
            if age_seconds < ttl_seconds:
                df_disk = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if not df_disk.empty:
                    return df_disk, None
        except Exception:
            pass
    try:
        if interval == "1m":
            df = _fetch_max_1m_stitched(ticker)
        else:
            try:
                df = _direct_yahoo_v8_chart(ticker, interval=interval, days_back=max_days)
            except Exception:
                t_obj = yf.Ticker(ticker)
                df = t_obj.history(period=f"{max_days}d", interval=interval, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].replace([np.inf, -np.inf], np.nan).dropna()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
        if not df.empty:
            df.to_csv(csv_path)
            return df, None
    except Exception:
        pass
    try:
        df_syn = _generate_intraday_from_daily(ticker, interval, max_days)
        return df_syn, f"Yahoo IP rate-limit active for '{interval}': Reconstructed max {max_days}d of {interval} bars anchored to real Daily OHLCV."
    except Exception as e:
        return pd.DataFrame(), f"Could not fetch or reconstruct {interval} data: {e}"

def fetch_and_construct_tf(ticker: str, tf: str, force_refresh: bool = False) -> tuple[pd.DataFrame, Optional[str]]:
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}_{tf}_fullmax"
    now_ts = datetime.now().timestamp()
    if tf in ["1Y", "1M", "1W", "1d"]:
        base_key = f"{ticker}_1d_fullmax"
        warn_msg = None
        if force_refresh or base_key not in DATA_CACHE:
            df_1d, warn_msg = _download_daily_max(ticker)
            if df_1d.empty:
                return pd.DataFrame(), warn_msg or f"No daily data found for {ticker}."
            DATA_CACHE[base_key] = df_1d
            DATA_CACHE_TS[base_key] = now_ts
        df_base = DATA_CACHE[base_key].copy()
        if tf == "1d":
            return df_base, warn_msg
        if tf == "1W":
            period_idx = df_base.index.to_period("W-SUN").start_time
        elif tf == "1M":
            period_idx = df_base.index.to_period("M").start_time
        else:
            period_idx = df_base.index.to_period("Y").start_time
        df_res = df_base.groupby(period_idx).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_res.index.name = "date"
        return df_res, warn_msg
    else:
        warn_msg = None
        cache_age = now_ts - DATA_CACHE_TS.get(cache_key, 0.0)
        ttl = 55.0 if tf == "1m" else 7200.0
        if force_refresh or (cache_key not in DATA_CACHE) or (cache_age >= ttl):
            df_intra, warn_msg = _download_intraday_max(ticker, interval=tf, force_refresh=force_refresh)
            if df_intra.empty:
                return pd.DataFrame(), warn_msg or f"No {tf} data found for {ticker}."
            DATA_CACHE[cache_key] = df_intra
            DATA_CACHE_TS[cache_key] = now_ts
        return DATA_CACHE[cache_key].copy(), warn_msg

# ============================================================================== #
# 2. QUANTITATIVE INDICATORS & DYNAMIC SL/TP FORMULA ENGINE
# ============================================================================== #
def compute_all_indicators(
    df: pd.DataFrame, tf: str, atr_mult: float = 1.5, gk_mult: float = 3.0,
    dynamic_tpsl: bool = True, w_trend: float = 0.4, w_liq: float = 0.4, w_rev: float = 0.2
) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    n = len(df)
    unavailable_indicators = []
    if n == 0:
        return df, ["All Indicators (Empty Data)"]
    if n < 20:
        unavailable_indicators.append("EMA 20 / Bollinger / GK Cone (< 20 bars in view)")
    if n < 50:
        unavailable_indicators.append("EMA 50 (< 50 bars in view)")
    if n < 200:
        unavailable_indicators.append("EMA 200 (< 200 bars in view; using adaptive span)")

    df['ema20'] = df['close'].ewm(span=20, min_periods=1).mean().bfill().fillna(df['close'])
    df['ema50'] = df['close'].ewm(span=50, min_periods=1).mean().bfill().fillna(df['close'])
    df['ema200'] = df['close'].ewm(span=200, min_periods=1).mean().bfill().fillna(df['close'])

    high_low = (df['high'] - df['low']).abs()
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14, min_periods=1).mean().bfill().fillna(df['close'] * 0.015)
    df['atr'] = np.where(df['atr'] <= 0, df['close'] * 0.015, df['atr'])

    safe_low = df['low'].replace(0, np.nan).bfill().fillna(1.0)
    safe_open = df['open'].replace(0, np.nan).bfill().fillna(1.0)
    log_hl = np.log((df['high'] / safe_low).clip(lower=1e-6))
    log_co = np.log((df['close'] / safe_open).clip(lower=1e-6))
    gk_var = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).clip(lower=1e-8)
    df['gk_vol_pct'] = np.sqrt(gk_var.ewm(span=20, min_periods=1).mean()).bfill().fillna(0.02)
    df['gk_sigma_dollar'] = (df['gk_vol_pct'] * df['close']).bfill().fillna(df['atr'])
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean().replace(0, 1.0).fillna(1.0)
    df['rvol'] = (df['volume'] / df['vol_sma20']).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = typical_price * df['volume']
    if tf in ["15m", "5m", "1m"]:
        dates = df.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = df['volume'].groupby(dates).cumsum().replace(0, 1.0)
        df['vwap'] = (cum_tp_vol / cum_vol).bfill().fillna(df['close'])
        std_vwap = typical_price.groupby(dates).transform(lambda x: x.expanding().std()).bfill().fillna(df['atr'] * 0.5)
    else:
        denom = df['volume'].rolling(20, min_periods=1).sum().replace(0, 1.0)
        df['vwap'] = (tp_vol.rolling(20, min_periods=1).sum() / denom).bfill().fillna(df['close'])
        std_vwap = df['close'].rolling(20, min_periods=1).std().bfill().fillna(df['atr'] * 0.5)
    std_vwap = np.where(std_vwap <= 0, df['atr'] * 0.5, std_vwap)
    df['vwap_u2'] = df['vwap'] + 2.0 * std_vwap
    df['vwap_l2'] = df['vwap'] - 2.0 * std_vwap

    delta = df['close'].diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['rsi'] = (100.0 - (100.0 / (1.0 + rs))).bfill().fillna(50.0)

    swing_window = min(50, max(5, n))
    df['swing_high'] = df['high'].rolling(swing_window, min_periods=1).max().bfill().fillna(df['high'])
    df['swing_low'] = df['low'].rolling(swing_window, min_periods=1).min().bfill().fillna(df['low'])
    fib_range = (df['swing_high'] - df['swing_low']).replace(0, 1e-6)
    df['fib_382'] = df['swing_high'] - 0.382 * fib_range
    df['fib_500'] = df['swing_high'] - 0.500 * fib_range
    df['fib_618'] = df['swing_high'] - 0.618 * fib_range
    df['fib_786'] = df['swing_high'] - 0.786 * fib_range

    body_prev = (df['close'].shift(1) - df['open'].shift(1)).abs().fillna(0.0)
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (body_prev > 0.8 * df['atr'])
    df['fvg_bull_top'] = np.where(df['fvg_bull'], df['low'], np.nan)
    df['fvg_bull_bot'] = np.where(df['fvg_bull'], df['high'].shift(2), np.nan)
    df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (body_prev > 0.8 * df['atr'])
    df['fvg_bear_top'] = np.where(df['fvg_bear'], df['low'].shift(2), np.nan)
    df['fvg_bear_bot'] = np.where(df['fvg_bear'], df['high'], np.nan)

    active_bull_top, active_bull_bot = np.nan, np.nan
    active_bear_top, active_bear_bot = np.nan, np.nan
    fvg_bull_retest = np.zeros(n, dtype=bool)
    fvg_bear_retest = np.zeros(n, dtype=bool)
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
    f_bull, f_bear = df['fvg_bull'].values, df['fvg_bear'].values
    for i in range(2, n):
        if not np.isnan(active_bull_top):
            if lows[i] <= active_bull_top and closes[i] >= active_bull_bot:
                fvg_bull_retest[i] = True
                active_bull_top = np.nan
            elif closes[i] < active_bull_bot:
                active_bull_top = np.nan
        if f_bull[i]:
            active_bull_top = lows[i]
            active_bull_bot = highs[i - 2]

        if not np.isnan(active_bear_bot):
            if highs[i] >= active_bear_bot and closes[i] <= active_bear_top:
                fvg_bear_retest[i] = True
                active_bear_bot = np.nan
            elif closes[i] > active_bear_top:
                active_bear_bot = np.nan
        if f_bear[i]:
            active_bear_top = lows[i - 2]
            active_bear_bot = highs[i]

    df['smc_fvg_bull_entry'] = fvg_bull_retest
    df['smc_fvg_bear_entry'] = fvg_bear_retest

    df['orb_high'] = np.nan
    df['orb_low'] = np.nan
    df['orb_mid'] = np.nan
    if tf in ["15m", "5m", "1m"]:
        for dt, group in df.groupby(df.index.date):
            start_t = group.index[0]
            orb_end_t = start_t + timedelta(minutes=30)
            orb_slice = group[group.index < orb_end_t]
            if not orb_slice.empty:
                oh = orb_slice['high'].max()
                ol = orb_slice['low'].min()
                om = (oh + ol) / 2.0
                df.loc[group.index, 'orb_high'] = oh
                df.loc[group.index, 'orb_low'] = ol
                df.loc[group.index, 'orb_mid'] = om
    else:
        df['orb_high'] = df['high'].rolling(5, min_periods=1).max().shift(1).bfill().fillna(df['high'])
        df['orb_low'] = df['low'].rolling(5, min_periods=1).min().shift(1).bfill().fillna(df['low'])
        df['orb_mid'] = (df['orb_high'] + df['orb_low']) / 2.0

    slope_norm = np.clip(((df['ema20'] - df['ema50']) / df['atr']), -2.0, 2.0) / 2.0
    bar_press = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)) * np.clip(df['rvol'] / 2.0, 0, 2.0)
    liq_norm = np.clip(bar_press, -1.0, 1.0)
    vwap_z = (df['close'] - df['vwap']) / std_vwap
    rev_norm = np.clip(-vwap_z / 2.5, -1.0, 1.0)
    total_w = abs(w_trend) + abs(w_liq) + abs(w_rev)
    if total_w == 0:
        total_w = 1.0
    df['composite_score'] = ((w_trend * slope_norm + w_liq * liq_norm + w_rev * rev_norm) / total_w).fillna(0.0)

    ema_diff_ratio = (df['ema20'] - df['ema50']).abs() / df['atr']
    df['is_unclear_volatile'] = (ema_diff_ratio < 0.32) | (
        (df['atr'] > 1.35 * df['atr'].rolling(20, min_periods=1).mean()) & (df['composite_score'].abs() < 0.20)
    )

    gk_sma20 = df['gk_vol_pct'].rolling(20, min_periods=1).mean().replace(0, 0.02).fillna(0.02)
    vol_ratio = (df['gk_vol_pct'] / gk_sma20).clip(0.65, 1.85)
    trend_strength = ema_diff_ratio.clip(0.0, 3.0)
    rvol_boost = (df['rvol'] - 1.0).clip(-0.4, 1.5)
    chop_penalty = np.where(df['is_unclear_volatile'], 1.12, 1.0)

    if dynamic_tpsl:
        dyn_sl = (atr_mult * np.sqrt(vol_ratio) * chop_penalty).clip(0.75, 3.50)
        dyn_tp_raw = (gk_mult * (0.75 + 0.35 * trend_strength + 0.20 * rvol_boost) * (1.0 + 0.25 * df['composite_score'].abs()))
        dyn_tp = np.maximum(dyn_sl * 1.25, dyn_tp_raw).clip(1.10, 7.50)
        df['dyn_sl_mult'] = dyn_sl.bfill().fillna(atr_mult)
        df['dyn_tp_mult'] = dyn_tp.bfill().fillna(gk_mult)
    else:
        df['dyn_sl_mult'] = float(atr_mult)
        df['dyn_tp_mult'] = float(gk_mult)

    drift = ((df['close'] - df['close'].shift(3)).fillna(0.0)) / 3.0
    df['gk_upper'] = df['close'] + drift + (df['dyn_tp_mult'] * df['gk_sigma_dollar'])
    df['gk_lower'] = df['close'] + drift - (df['dyn_tp_mult'] * df['gk_sigma_dollar'])
    df['atr_upper'] = df['close'] + (df['dyn_sl_mult'] * df['atr'])
    df['atr_lower'] = df['close'] - (df['dyn_sl_mult'] * df['atr'])
    return df, unavailable_indicators

# ============================================================================== #
# 3. STRATEGIES, RULES & BACKTEST ENGINE
# ============================================================================== #
STRATEGY_METADATA = {
    "trend": {
        "name": "1. Core Trend (EMA 20/50/200 + Volume)",
        "description": "Rides established intermediate trends using EMA crossovers and volume confirmation.",
        "indicators": ["ema20", "ema50", "ema200", "atr_bands"],
        "bucket": "Core 70%"
    },
    "watch_dip": {
        "name": "2. Watch -> Pullback/Rally -> Buy/Short",
        "description": "Monitors market swings to enter on constructive pullbacks in bull trends or rallies in bear trends.",
        "indicators": ["ema20", "ema50", "ema200", "vwap", "gk_bands"],
        "bucket": "Core 70%"
    },
    "smc_fvg": {
        "name": "3. SMC & Fair Value Gaps (Bull/Bear FVG)",
        "description": "Trades institutional liquidity imbalances and fair value gap retest zones.",
        "indicators": ["fvg_zones", "ema50", "ema200", "atr_bands"],
        "bucket": "Satellite 30%"
    },
    "fib_golden": {
        "name": "4. Fibonacci Golden Pocket (0.618 / 0.382)",
        "description": "Seeks high-probability reversals at key Fibonacci retracement levels combined with RSI filters.",
        "indicators": ["fib_levels", "ema50", "ema200"],
        "bucket": "Core 70%"
    },
    "orb": {
        "name": "5. Opening Range Breakout/Breakdown (ORB)",
        "description": "Captures momentum surges breaking beyond the initial session range with high relative volume.",
        "indicators": ["orb_levels", "vwap", "ema200"],
        "bucket": "Satellite 30%"
    },
    "do_t": {
        "name": "6. Intraday 'Do T' (VWAP +-2 Sigma L/S)",
        "description": "Mean-reversion strategy fading extreme statistical deviations around VWAP bands.",
        "indicators": ["vwap", "vwap_bands", "gk_bands", "ema200"],
        "bucket": "Satellite 30%"
    },
    "composite": {
        "name": "7. Composite Score (Alpha*A + Beta*B + Gamma*C)",
        "description": "Multi-factor scoring weighting trend slope, bar pressure, and VWAP distance.",
        "indicators": ["ema20", "ema200", "vwap", "gk_bands"],
        "bucket": "Core 70%"
    },
    "scale_ladder": {
        "name": "8. Momentum Scale-Out Ladder (Sell xx% @ +xx% Tiers)",
        "description": "Scales out partial profits progressively at tiered gain thresholds while locking in breakeven stops.",
        "indicators": ["ema20", "ema200", "vwap", "gk_bands", "atr_bands"],
        "bucket": "Core 70%"
    },
    "plain_tpsl": {
        "name": "9. Plain TP/SL & Manual Replay Sandbox",
        "description": "Interactive manual execution sandbox with fixed or dynamic risk-reward targets.",
        "indicators": ["ema20", "ema200", "atr_bands", "gk_bands"],
        "bucket": "Sandbox"
    }
}

def _fmt_dt(dt_obj: pd.Timestamp, tf: str) -> str:
    if tf in ["1h", "15m", "5m", "1m"]:
        return dt_obj.strftime("%Y-%m-%d %H:%M")
    return dt_obj.strftime("%Y-%m-%d")

def run_strategy_backtest(
    df: pd.DataFrame, tf: str, strategy_key: str, trade_side: str = "both",
    ema200_filter: str = "none", sl_mult: float = 1.5, tp_mult: float = 3.0,
    vol_model: str = "atr", enable_scale_out: bool = True,
    enable_trailing_only: bool = False,
    t1_gain_pct: float = 4.0, t1_sell_pct: float = 35.0,
    t2_gain_pct: float = 8.0, t2_sell_pct: float = 35.0,
    initial_capital: float = 100000.0, core_pct: float = 0.70,
    intraday_pct: float = 0.30, risk_per_trade_pct: float = 0.01,
    manual_entry_ts: Optional[int] = None, manual_side: str = "long",
    manual_alloc_pct: Optional[float] = None
) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"trades": [], "markers": [], "boxes": [], "stats": {}, "equity": []}
    allow_long = trade_side in ["both", "long"]
    allow_short = trade_side in ["both", "short"]
    dt_index = df.index
    timestamps = [int(ts.timestamp()) for ts in dt_index]
    opens, highs, lows, closes, volumes = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    atrs, gks = df['atr'].values, df['gk_sigma_dollar'].values
    dyn_sl_arr, dyn_tp_arr = df['dyn_sl_mult'].values, df['dyn_tp_mult'].values
    ema20, ema50, ema200 = df['ema20'].values, df['ema50'].values, df['ema200'].values
    rvol, vwap, vwap_l2, vwap_u2 = df['rvol'].values, df['vwap'].values, df['vwap_l2'].values, df['vwap_u2'].values
    fib_382, fib_618, fib_786 = df['fib_382'].values, df['fib_618'].values, df['fib_786'].values
    swing_high = df['swing_high'].values
    orb_high, orb_low, orb_mid = df['orb_high'].values, df['orb_low'].values, df['orb_mid'].values
    smc_bull, smc_bear = df['smc_fvg_bull_entry'].values, df['smc_fvg_bear_entry'].values
    comp_score, rsi = df['composite_score'].values, df['rsi'].values

    is_intraday_strat = strategy_key in ["orb", "do_t", "smc_fvg"]
    core_cash = initial_capital * core_pct
    intraday_cash = initial_capital * intraday_pct
    pool_label = f"Intraday ({int(intraday_pct*100)}%)" if is_intraday_strat else f"Core ({int(core_pct*100)}%)"
    peak_equity = initial_capital
    max_dd_pct = 0.0

    state = 0
    watch_peak, watch_trough = 0.0, 0.0
    entry_price, sl_price, initial_sl_price, tp_price = 0.0, 0.0, 0.0, 0.0
    alt_sl_price, alt_tp_price = 0.0, 0.0
    entry_sl_mult_used, entry_tp_mult_used = sl_mult, tp_mult
    entry_idx = 0
    initial_shares, remaining_shares = 0, 0
    reserved_cash, realized_trade_pnl = 0.0, 0.0
    scale_stage = 0
    t1_target_price, t2_target_price = 0.0, 0.0
    scale_notes: List[str] = []
    entry_signal_reason = ""
    dollar_risk_at_entry = 0.0
    trades, markers, boxes, equity_curve = [], [], [], []

    def _get_active_pool_cash() -> float:
        return intraday_cash if is_intraday_strat else core_cash

    def _set_active_pool_cash(new_val: float) -> None:
        nonlocal core_cash, intraday_cash
        if is_intraday_strat:
            intraday_cash = new_val
        else:
            core_cash = new_val

    def _total_portfolio_equity(mark_price: float) -> float:
        open_val = 0.0
        if state == 2 and remaining_shares > 0:
            open_val = remaining_shares * mark_price
        elif state == -2 and remaining_shares > 0:
            open_val = reserved_cash + (entry_price - mark_price) * remaining_shares
        return core_cash + intraday_cash + open_val

    def _calc_shares(ep: float, slp: float, force_alloc_pct: Optional[float] = None) -> tuple[int, float]:
        tot_eq = max(_total_portfolio_equity(ep), 1000.0)
        avail_pool_cash = max(_get_active_pool_cash(), 0.0)
        if force_alloc_pct is not None and force_alloc_pct > 0:
            target_dollars = tot_eq * (min(force_alloc_pct, 100.0) / 100.0)
            shares_int = max(1, int(math.floor(target_dollars / max(ep, 0.01))))
            actual_risk = shares_int * abs(ep - slp)
            return shares_int, actual_risk
        if avail_pool_cash < ep:
            return 0, 0.0
        risk_budget = tot_eq * risk_per_trade_pct
        stop_dist = max(abs(ep - slp), ep * 0.004, 0.01)
        raw_shares = min(risk_budget / stop_dist, (avail_pool_cash * 0.98) / max(ep, 0.01))
        if math.isnan(raw_shares) or math.isinf(raw_shares):
            raw_shares = 0.0
        shares_int = max(0, int(math.floor(raw_shares)))
        if shares_int == 0 and avail_pool_cash >= ep:
            shares_int = 1
        actual_risk = shares_int * abs(ep - slp)
        return shares_int, actual_risk

    def _compute_tier_targets(ep: float, tp_p: float, is_long: bool) -> tuple[float, float]:
        full_dist = abs(tp_p - ep)
        pct_dist_1 = ep * (max(t1_gain_pct, 0.1) / 100.0)
        pct_dist_2 = ep * (max(t2_gain_pct, 0.2) / 100.0)
        d1 = min(pct_dist_1, full_dist * 0.45) if tf in ["1h", "15m", "5m", "1m"] else min(pct_dist_1, full_dist * 0.55)
        d2 = min(pct_dist_2, full_dist * 0.78) if tf in ["1h", "15m", "5m", "1m"] else min(pct_dist_2, full_dist * 0.82)
        d1 = max(d1, ep * 0.002)
        d2 = max(d2, d1 * 1.35)
        if is_long:
            return ep + d1, ep + d2
        else:
            return ep - d1, ep - d2

    for i in range(1, n):
        ts = timestamps[i]
        c_high, c_low, c_close, c_open = highs[i], lows[i], closes[i], opens[i]
        unit_vol = gks[i] if vol_model == "gk" else atrs[i]
        if np.isnan(unit_vol) or math.isinf(unit_vol) or unit_vol <= 0:
            unit_vol = max(c_close * 0.015, 0.01)
        cur_sl_mult = _clean_float(dyn_sl_arr[i], sl_mult, 2)
        cur_tp_mult = _clean_float(dyn_tp_arr[i], tp_mult, 2)

        trend_long, trend_short = False, False
        trig_long, trig_short = False, False

        if strategy_key == "trend":
            trend_long = c_close > ema50[i] and ema20[i] > ema50[i]
            trig_long = trend_long and (c_low <= ema20[i] * 1.008) and (c_close > c_open) and (rvol[i] >= 0.9)
            trend_short = c_close < ema50[i] and ema20[i] < ema50[i]
            trig_short = trend_short and (c_high >= ema20[i] * 0.992) and (c_close < c_open) and (rvol[i] >= 0.9)
        elif strategy_key == "watch_dip":
            trend_long = ema20[i] > ema50[i]
            trend_short = ema20[i] < ema50[i]
            if state in [1, 3]:
                watch_peak = max(watch_peak, c_high)
                trig_long = ((watch_peak - c_low) >= 0.8 * unit_vol) and (c_close > c_open)
            if state in [-1, -3]:
                watch_trough = min(watch_trough, c_low) if watch_trough > 0 else c_low
                trig_short = ((c_high - watch_trough) >= 0.8 * unit_vol) and (c_close < c_open)
        elif strategy_key == "smc_fvg":
            trend_long = c_close >= ema50[i] * 0.98
            trig_long = bool(smc_bull[i])
            trend_short = c_close <= ema50[i] * 1.02
            trig_short = bool(smc_bear[i])
        elif strategy_key == "fib_golden":
            trend_long = c_close > fib_786[i] and ema20[i] >= ema50[i] * 0.99
            trig_long = trend_long and (c_low <= fib_618[i] * 1.005) and (c_close >= fib_618[i]) and (rsi[i] < 60)
            trend_short = c_close < swing_high[i] and ema20[i] < ema50[i] * 1.01
            trig_short = trend_short and (c_high >= fib_382[i] * 0.995) and (c_close <= fib_382[i]) and (rsi[i] > 40)
        elif strategy_key == "orb":
            if not np.isnan(orb_high[i]) and not np.isnan(orb_low[i]):
                trend_long = c_close > orb_mid[i] and c_close > vwap[i]
                trig_long = (closes[i - 1] <= orb_high[i - 1]) and (c_close > orb_high[i]) and (rvol[i] >= 1.2)
                trend_short = c_close < orb_mid[i] and c_close < vwap[i]
                trig_short = (closes[i - 1] >= orb_low[i - 1]) and (c_close < orb_low[i]) and (rvol[i] >= 1.2)
        elif strategy_key == "do_t":
            trend_long = c_close > ema50[i] * 0.93
            trig_long = (c_low <= vwap_l2[i]) and (c_close > c_open)
            trend_short = c_close < ema50[i] * 1.07
            trig_short = (c_high >= vwap_u2[i]) and (c_close < c_open)
        elif strategy_key == "composite":
            trend_long = comp_score[i] > 0.15
            trig_long = comp_score[i] >= 0.48
            trend_short = comp_score[i] < -0.15
            trig_short = comp_score[i] <= -0.48
        elif strategy_key == "scale_ladder":
            trend_long = (c_close > ema20[i]) and (ema20[i] >= ema50[i] * 0.995)
            trig_long = trend_long and (c_close > vwap[i]) and (rvol[i] >= 1.05) and (c_close > c_open)
            trend_short = (c_close < ema20[i]) and (ema20[i] <= ema50[i] * 1.005)
            trig_short = trend_short and (c_close < vwap[i]) and (rvol[i] >= 1.05) and (c_close < c_open)
        elif strategy_key == "plain_tpsl":
            if manual_entry_ts is not None:
                is_hit_bar = (ts >= manual_entry_ts) and (timestamps[i - 1] < manual_entry_ts or i == 1)
                if manual_side == "short":
                    trend_short, trig_short = True, is_hit_bar
                else:
                    trend_long, trig_long = True, is_hit_bar
            else:
                trend_long = c_close > ema20[i]
                trig_long = (c_close > highs[i - 1]) and (c_close > c_open)
                trend_short = c_close < ema20[i]
                trig_short = (c_close < lows[i - 1]) and (c_close < c_open)

        if manual_entry_ts is None:
            if ema200_filter == "trend_follow":
                if c_close < ema200[i]:
                    trend_long, trig_long = False, False
                if c_close > ema200[i]:
                    trend_short, trig_short = False, False
            elif ema200_filter == "counter_trend":
                if c_close >= ema200[i]:
                    trend_long, trig_long = False, False
                if c_close <= ema200[i]:
                    trend_short, trig_short = False, False

        if not allow_long:
            trend_long, trig_long = False, False
        if not allow_short:
            trend_short, trig_short = False, False

        # State machine
        if state == 0:
            if strategy_key == "plain_tpsl" and manual_entry_ts is not None and (trig_long or trig_short):
                is_long = trig_long
                entry_price, entry_idx = c_close, i
                entry_sl_mult_used, entry_tp_mult_used = cur_sl_mult, cur_tp_mult
                safe_atr = _clean_float(atrs[i], c_close * 0.015)
                safe_gk = _clean_float(gks[i], safe_atr)
                if is_long:
                    sl_price = entry_price - cur_sl_mult * safe_atr
                    tp_price = entry_price + cur_tp_mult * safe_atr
                else:
                    sl_price = entry_price + cur_sl_mult * safe_atr
                    tp_price = entry_price - cur_tp_mult * safe_atr
                sh, d_risk = _calc_shares(entry_price, sl_price, force_alloc_pct=manual_alloc_pct)
                if sh > 0:
                    state = 2 if is_long else -2
                    initial_shares, remaining_shares = sh, sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage, scale_notes = 0, []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, is_long)
                    entry_signal_reason = f"MANUAL (Trailing Only)" if enable_trailing_only else f"MANUAL ({cur_sl_mult}x)"
                    markers.append({
                        "time": ts, "position": "belowBar" if is_long else "aboveBar",
                        "color": "#00E676" if is_long else "#FF4081",
                        "shape": "arrowUp" if is_long else "arrowDown",
                        "text": f"MANUAL {'BUY' if is_long else 'SHORT'} ({sh} sh)"
                    })
            elif trend_long:
                state = 1
                watch_peak = c_high
                markers.append({"time": ts, "position": "belowBar", "color": "#90A4AE", "shape": "circle", "text": "WATCH L"})
            elif trend_short:
                state = -1
                watch_trough = c_low
                markers.append({"time": ts, "position": "aboveBar", "color": "#B0BEC5", "shape": "circle", "text": "WATCH S"})
        elif state in [1, 3]:
            if not trend_long and manual_entry_ts is None:
                state = -1 if trend_short else 0
                watch_trough = c_low
            elif trig_long:
                is_re = (state == 3)
                entry_price, entry_idx = c_close, i
                entry_sl_mult_used, entry_tp_mult_used = cur_sl_mult, cur_tp_mult
                sl_price = entry_price - cur_sl_mult * unit_vol
                tp_price = entry_price + cur_tp_mult * unit_vol
                sh, d_risk = _calc_shares(entry_price, sl_price)
                if sh > 0:
                    state = 2
                    initial_shares, remaining_shares = sh, sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage, scale_notes = 0, []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, True)
                    entry_signal_reason = f"{'RE-LONG' if is_re else 'LONG'} (Trailing Only)" if enable_trailing_only else f"LONG"
                    markers.append({"time": ts, "position": "belowBar", "color": "#00E676", "shape": "arrowUp", "text": f"BUY {sh} sh"})
        elif state in [-1, -3]:
            if not trend_short and manual_entry_ts is None:
                state = 1 if trend_long else 0
                watch_peak = c_high
            elif trig_short:
                is_re = (state == -3)
                entry_price, entry_idx = c_close, i
                entry_sl_mult_used, entry_tp_mult_used = cur_sl_mult, cur_tp_mult
                sl_price = entry_price + cur_sl_mult * unit_vol
                tp_price = entry_price - cur_tp_mult * unit_vol
                sh, d_risk = _calc_shares(entry_price, sl_price)
                if sh > 0:
                    state = -2
                    initial_shares, remaining_shares = sh, sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage, scale_notes = 0, []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, False)
                    entry_signal_reason = f"{'RE-SHORT' if is_re else 'SHORT'} (Trailing Only)" if enable_trailing_only else f"SHORT"
                    markers.append({"time": ts, "position": "aboveBar", "color": "#FF1744", "shape": "arrowDown", "text": f"SHORT {sh} sh"})

        # Active Long Management with Trailing Rule Support
        elif state == 2:
            if enable_trailing_only:
                suggested_trail = c_close - (cur_sl_mult * unit_vol)
                sl_price = max(sl_price, suggested_trail)
                hit_tp = False
                hit_sl = c_low <= sl_price
                trend_flip_exit = trend_short
            else:
                hit_sl = c_low <= sl_price
                hit_tp = c_high >= tp_price
                trend_flip_exit = False

            if hit_tp or hit_sl or trend_flip_exit:
                exit_price = tp_price if hit_tp else (sl_price if hit_sl else c_close)
                rem_pnl = (exit_price - entry_price) * remaining_shares
                _set_active_pool_cash(_get_active_pool_cash() + (remaining_shares * entry_price) + rem_pnl)
                total_pnl = realized_trade_pnl + rem_pnl
                pos_val = entry_price * initial_shares
                ret_pct = (total_pnl / max(pos_val, 1e-6)) * 100.0
                remaining_shares, reserved_cash = 0, 0.0
                state = 0
                cur_eq = _total_portfolio_equity(c_close)
                alloc_pct = (pos_val / max(cur_eq - total_pnl, 1.0)) * 100.0
                res_type = "TREND-FLIP" if trend_flip_exit else ("TP" if hit_tp else ("TRAIL-WIN" if total_pnl > 0 else "SL"))

                trades.append({
                    "trade_no": len(trades) + 1, "side": "LONG", "signal_type": entry_signal_reason,
                    "pool": pool_label, "entry_bar_idx": int(entry_idx), "exit_bar_idx": int(i),
                    "entry_time": timestamps[entry_idx], "entry_dt": _fmt_dt(dt_index[entry_idx], tf),
                    "exit_time": ts, "exit_dt": _fmt_dt(dt_index[i], tf), "type": res_type,
                    "entry_price": _clean_float(entry_price, 0.0, 2), "exit_price": _clean_float(exit_price, 0.0, 2),
                    "sl_price": _clean_float(initial_sl_price, 0.0, 2), "tp_price": _clean_float(tp_price, 0.0, 2) if not enable_trailing_only else "None (Trailing)",
                    "dyn_mults": f"{entry_sl_mult_used:.2f}x", "scale_summary": "Trailing Stop / No TP",
                    "shares": int(initial_shares), "position_value": _clean_float(pos_val, 0.0, 2),
                    "alloc_pct": _clean_float(alloc_pct, 0.0, 1), "dollar_risk": _clean_float(dollar_risk_at_entry, 0.0, 2),
                    "bar_volume": int(_clean_float(volumes[entry_idx], 0.0)), "bar_rvol": _clean_float(rvol[entry_idx], 1.0, 2),
                    "pnl": _clean_float(total_pnl, 0.0, 2), "ret_pct": _clean_float(ret_pct, 0.0, 2),
                    "equity_after": _clean_float(cur_eq, initial_capital, 2)
                })
                boxes.append({
                    "trade_no": len(trades), "side": "LONG", "t1": timestamps[entry_idx], "t2": ts,
                    "entry": _clean_float(entry_price), "tp": _clean_float(tp_price) if not enable_trailing_only else _clean_float(c_close), "sl": _clean_float(initial_sl_price),
                    "outcome": res_type
                })
                markers.append({"time": ts, "position": "aboveBar", "color": "#FFD600" if total_pnl > 0 else "#FF1744", "shape": "arrowDown", "text": f"L-{res_type} (${total_pnl:,.0f})"})
                state = -1 if trend_short else (1 if trend_long else 0)

        # Active Short Management with Trailing Rule Support
        elif state == -2:
            if enable_trailing_only:
                suggested_trail = c_close + (cur_sl_mult * unit_vol)
                sl_price = min(sl_price, suggested_trail)
                hit_tp = False
                hit_sl = c_high >= sl_price
                trend_flip_exit = trend_long
            else:
                hit_sl = c_high >= sl_price
                hit_tp = c_low <= tp_price
                trend_flip_exit = False

            if hit_tp or hit_sl or trend_flip_exit:
                exit_price = tp_price if hit_tp else (sl_price if hit_sl else c_close)
                rem_pnl = (entry_price - exit_price) * remaining_shares
                _set_active_pool_cash(_get_active_pool_cash() + (remaining_shares * entry_price) + rem_pnl)
                total_pnl = realized_trade_pnl + rem_pnl
                pos_val = entry_price * initial_shares
                ret_pct = (total_pnl / max(pos_val, 1e-6)) * 100.0
                remaining_shares, reserved_cash = 0, 0.0
                state = 0
                cur_eq = _total_portfolio_equity(c_close)
                alloc_pct = (pos_val / max(cur_eq - total_pnl, 1.0)) * 100.0
                res_type = "TREND-FLIP" if trend_flip_exit else ("TP" if hit_tp else ("TRAIL-WIN" if total_pnl > 0 else "SL"))

                trades.append({
                    "trade_no": len(trades) + 1, "side": "SHORT", "signal_type": entry_signal_reason,
                    "pool": pool_label, "entry_bar_idx": int(entry_idx), "exit_bar_idx": int(i),
                    "entry_time": timestamps[entry_idx], "entry_dt": _fmt_dt(dt_index[entry_idx], tf),
                    "exit_time": ts, "exit_dt": _fmt_dt(dt_index[i], tf), "type": res_type,
                    "entry_price": _clean_float(entry_price, 0.0, 2), "exit_price": _clean_float(exit_price, 0.0, 2),
                    "sl_price": _clean_float(initial_sl_price, 0.0, 2), "tp_price": _clean_float(tp_price, 0.0, 2) if not enable_trailing_only else "None (Trailing)",
                    "dyn_mults": f"{entry_sl_mult_used:.2f}x", "scale_summary": "Trailing Stop / No TP",
                    "shares": int(initial_shares), "position_value": _clean_float(pos_val, 0.0, 2),
                    "alloc_pct": _clean_float(alloc_pct, 0.0, 1), "dollar_risk": _clean_float(dollar_risk_at_entry, 0.0, 2),
                    "bar_volume": int(_clean_float(volumes[entry_idx], 0.0)), "bar_rvol": _clean_float(rvol[entry_idx], 1.0, 2),
                    "pnl": _clean_float(total_pnl, 0.0, 2), "ret_pct": _clean_float(ret_pct, 0.0, 2),
                    "equity_after": _clean_float(cur_eq, initial_capital, 2)
                })
                boxes.append({
                    "trade_no": len(trades), "side": "SHORT", "t1": timestamps[entry_idx], "t2": ts,
                    "entry": _clean_float(entry_price), "tp": _clean_float(tp_price) if not enable_trailing_only else _clean_float(c_close), "sl": _clean_float(initial_sl_price),
                    "outcome": res_type
                })
                markers.append({"time": ts, "position": "belowBar", "color": "#FFD600" if total_pnl > 0 else "#FF1744", "shape": "arrowUp", "text": f"S-{res_type} (${total_pnl:,.0f})"})
                state = -1 if trend_short else (1 if trend_long else 0)

        live_eq = _total_portfolio_equity(c_close)
        peak_equity = max(peak_equity, live_eq)
        dd = ((peak_equity - live_eq) / max(peak_equity, 1.0)) * 100.0
        max_dd_pct = max(max_dd_pct, dd)
        equity_curve.append({"time": ts, "value": _clean_float(live_eq, initial_capital, 2)})

    final_mark_equity = _total_portfolio_equity(closes[-1]) if n > 0 else initial_capital
    closed_trades = [t for t in trades if t["type"] in ["TP", "TRAIL-WIN", "SL", "TREND-FLIP", "MANUAL-WIN", "MANUAL-EXIT"]]
    total_trades = len(closed_trades)
    long_trades = sum(1 for t in closed_trades if t.get("side") == "LONG")
    short_trades = sum(1 for t in closed_trades if t.get("side") == "SHORT")
    wins = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] <= 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    net_return_pct = ((final_mark_equity - initial_capital) / max(initial_capital, 1.0)) * 100.0
    buy_hold_ret = ((closes[-1] - closes[0]) / max(closes[0], 1e-6)) * 100.0 if n > 1 else 0.0

    stats = {
        "strategy_key": strategy_key, "strategy_name": STRATEGY_METADATA[strategy_key]["name"],
        "bucket": STRATEGY_METADATA[strategy_key]["bucket"],
        "initial_capital": _clean_float(initial_capital, 100000.0, 2),
        "ending_equity": _clean_float(final_mark_equity, initial_capital, 2),
        "net_pnl": _clean_float(final_mark_equity - initial_capital, 0.0, 2),
        "net_return_pct": _clean_float(net_return_pct, 0.0, 2),
        "buy_hold_pct": _clean_float(buy_hold_ret, 0.0, 2),
        "total_trades": total_trades, "long_trades": long_trades, "short_trades": short_trades,
        "win_rate_pct": _clean_float(win_rate, 0.0, 1),
        "profit_factor": _clean_float(min(profit_factor, 99.9), 0.0, 2),
        "max_dd_pct": _clean_float(max_dd_pct, 0.0, 2)
    }
    return {"trades": trades, "markers": markers, "boxes": boxes, "stats": stats, "equity": equity_curve}

# ============================================================================== #
# 4. BLACK-SCHOLES & FASTAPI ENDPOINTS
# ============================================================================== #
def black_scholes_analysis(S: float, K: float, days_to_expiry: int, iv_pct: float, r_pct: float, gk_daily_vol_pct: float, opt_type: str = "call"):
    S, K = max(_clean_float(S, 100.0), 0.01), max(_clean_float(K, 100.0), 0.01)
    T = max(_clean_float(days_to_expiry, 14) / 365.0, 1 / 3650.0)
    sigma = max(_clean_float(iv_pct, 50.0) / 100.0, 0.01)
    r = _clean_float(r_pct, 4.25) / 100.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    return _sanitize_for_json({"theo_price": _clean_float(price, 0.01, 3), "delta": _clean_float(delta, 0.0, 3)})

@app.get("/api/analyze")
def api_analyze(
    ticker: str = Query("SNDK"), tf: str = Query("1Y"), start_ts: Optional[int] = Query(None),
    cutoff_ts: Optional[int] = Query(None), strategy: str = Query("trend"), trade_side: str = Query("both"),
    ema200_filter: str = Query("none"), sl_mult: float = Query(1.5), tp_mult: float = Query(3.0),
    dynamic_tpsl: bool = Query(True), vol_model: str = Query("atr"), enable_scale_out: bool = Query(True),
    enable_trailing_only: bool = Query(False),
    initial_capital: float = Query(100000.0), core_pct: float = Query(70.0), intraday_pct: float = Query(30.0),
    risk_pct: float = Query(1.0), manual_entry_ts: Optional[int] = Query(None), manual_side: str = Query("long"),
    manual_alloc_pct: Optional[float] = Query(None), force_refresh: bool = Query(False)
):
    df_raw, warn_msg = fetch_and_construct_tf(ticker, tf, force_refresh=force_refresh)
    if df_raw.empty:
        return JSONResponse({"error": warn_msg or "No data available."}, status_code=400)

    df_ind, unavailable_list = compute_all_indicators(df_raw, tf, atr_mult=sl_mult, gk_mult=tp_mult, dynamic_tpsl=dynamic_tpsl)
    bt_selected = run_strategy_backtest(
        df_ind, tf=tf, strategy_key=strategy, trade_side=trade_side, ema200_filter=ema200_filter,
        sl_mult=sl_mult, tp_mult=tp_mult, vol_model=vol_model, enable_scale_out=enable_scale_out,
        enable_trailing_only=enable_trailing_only, initial_capital=initial_capital,
        core_pct=core_pct / 100.0, intraday_pct=intraday_pct / 100.0, risk_per_trade_pct=risk_pct / 100.0,
        manual_entry_ts=manual_entry_ts, manual_side=manual_side, manual_alloc_pct=manual_alloc_pct
    )

    all_summary = []
    for s_key in STRATEGY_METADATA.keys():
        res = run_strategy_backtest(
            df_ind, tf=tf, strategy_key=s_key, trade_side=trade_side, ema200_filter=ema200_filter,
            sl_mult=sl_mult, tp_mult=tp_mult, vol_model=vol_model, enable_scale_out=enable_scale_out,
            enable_trailing_only=enable_trailing_only, initial_capital=initial_capital,
            core_pct=core_pct / 100.0, intraday_pct=intraday_pct / 100.0, risk_per_trade_pct=risk_pct / 100.0
        )
        all_summary.append(res["stats"])

    candles, volumes, bar_details = [], [], {}
    series_data = {
        "ema20": [], "ema50": [], "ema200": [], "vwap": [], "vwap_u2": [], "vwap_l2": [],
        "atr_upper": [], "atr_lower": [], "gk_upper": [], "gk_lower": [],
        "fib_382": [], "fib_500": [], "fib_618": [], "fib_786": [], "orb_high": [], "orb_low": []
    }
    idx_timestamps = [int(t.timestamp()) for t in df_ind.index]

    for i, (dt, row) in enumerate(df_ind.iterrows()):
        ts = idx_timestamps[i]
        o_val, h_val, l_val, c_val = _clean_float(row['open']), _clean_float(row['high']), _clean_float(row['low']), _clean_float(row['close'])
        v_val = _clean_float(row['volume'])
        candles.append({"time": ts, "open": o_val, "high": h_val, "low": l_val, "close": c_val})
        volumes.append({"time": ts, "value": v_val, "color": "rgba(0, 230, 118, 0.35)" if c_val >= o_val else "rgba(255, 23, 68, 0.35)"})
        bar_details[str(ts)] = {
            "dt": _fmt_dt(dt, tf), "o": o_val, "h": h_val, "l": l_val, "c": c_val,
            "ema20": _clean_float(row['ema20']), "ema50": _clean_float(row['ema50']), "ema200": _clean_float(row['ema200'])
        }
        for col in series_data.keys():
            val = row.get(col, np.nan)
            if not pd.isna(val):
                series_data[col].append({"time": ts, "value": round(float(val), 4)})

    last_row = df_ind.iloc[-1]
    payload = {
        "ticker": ticker.upper(), "tf": tf, "total_bars": len(candles),
        "candles": candles, "volumes": volumes, "bar_details": bar_details, "series": series_data,
        "backtest": bt_selected, "all_strategies_summary": all_summary,
        "latest_metrics": {
            "close": _clean_float(last_row['close'], 100.0, 2), "ema200": _clean_float(last_row['ema200'], 100.0, 2),
            "atr": _clean_float(last_row['atr'], 1.5, 2), "gk_daily_vol_pct": _clean_float(float(last_row['gk_vol_pct']) * 100.0, 2.5, 2)
        }
    }
    return JSONResponse(content=_sanitize_for_json(payload))

# ============================================================================== #
# 6. FRONTEND SINGLE-PAGE APPLICATION
# ============================================================================== #
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quant Lab: Trailing Stop / No-TP Rule, Async Loading & Replay Shortcuts</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root { --bg: #0b0e14; --panel: #131722; --border: #2a2e39; --text: #d1d4dc; --muted: #787b86; --accent: #2962ff; --green: #00e676; --red: #ff1744; --gold: #ffd600; }
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; font-size: 13px; }
.sidebar { width: 355px; background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
details.fold-card { background: #1a1f2e; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }
details.fold-card > summary { font-weight: 700; font-size: 11.5px; text-transform: uppercase; color: #90a4ae; cursor: pointer; list-style: none; display: flex; justify-content: space-between; align-items: center; }
details.fold-card[open] > summary { margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 3px; margin-top: 6px; }
input[type="text"], input[type="number"], select { width: 100%; background: #0b0e14; border: 1px solid var(--border); color: var(--text); padding: 5px 8px; border-radius: 4px; font-size: 12px; }
.row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.chk-item { display: flex; align-items: center; gap: 6px; font-size: 11px; cursor: pointer; }
button.btn { background: var(--accent); color: #fff; border: none; border-radius: 4px; padding: 6px 10px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 8px; font-size: 12px; }
button.btn:hover { opacity: 0.9; }
.main { flex: 1; display: flex; flex-direction: column; height: 100vh; overflow-y: auto; }
.topbar { background: var(--panel); border-bottom: 1px solid var(--border); padding: 6px 12px; display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-shrink: 0; }
.tf-selector { display: flex; gap: 4px; align-items: center; }
.tf-btn { background: #1a1f2e; border: 1px solid var(--border); color: var(--muted); padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 600; }
.tf-btn.active { background: #00b0ff; color: #000; border-color: #00b0ff; }
.replay-bar { background: #161b28; border-bottom: 1px solid var(--border); padding: 6px 12px; display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-shrink: 0; }
.rp-btn { background: #23293a; border: 1px solid #3a4259; color: var(--text); padding: 4px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; }
.rp-btn.active { background: var(--gold); color: #000; border-color: var(--gold); }
.alert-banner { background: rgba(255, 152, 0, 0.15); border-bottom: 1px solid #ff9800; color: #ffb74d; padding: 5px 14px; font-size: 11.5px; display: none; flex-shrink: 0; }
.chart-wrapper { position: relative; width: 100%; flex: 1; min-height: 430px; background: #0b0e14; border-bottom: 1px solid var(--border); }
#tv-chart { width: 100%; height: 100%; }
.bottom-panels { padding: 8px 12px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th, td { padding: 5px 7px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); background: #161b26; position: sticky; top: 0; }
.pos { color: var(--green); } .neg { color: var(--red); }
</style>
</head>
<body>

<div class="sidebar" id="sidebar">
    <details class="fold-card" open>
        <summary>1. Asset & Timeframe</summary>
        <label>Ticker</label><input type="text" id="ticker" value="SNDK" onchange="loadChartData()">
        <label>Trade Direction</label>
        <select id="trade_side" onchange="loadChartData()">
            <option value="both" selected>Long & Short (Both)</option>
            <option value="long">Long Only</option>
            <option value="short">Short Only</option>
        </select>
    </details>

    <details class="fold-card" open>
        <summary>2. Strategy & Rule Engine</summary>
        <div id="strategy-guide-content" style="font-size:11px; line-height:1.4; color:var(--muted); max-height:120px; overflow-y:auto; margin-bottom:6px;"></div>
        <label>Active Strategy</label>
        <select id="strategy" onchange="loadChartData()">
            <option value="trend">1. Core Trend (EMA 20/50/200 + Volume L/S)</option>
            <option value="watch_dip">2. Watch -> Pullback/Rally -> Buy/Short</option>
            <option value="smc_fvg">3. SMC & Fair Value Gaps (Bull/Bear FVG)</option>
            <option value="fib_golden">4. Fibonacci Golden Pocket (0.618 / 0.382)</option>
            <option value="orb">5. Opening Range Breakout (ORB)</option>
            <option value="do_t">6. Intraday 'Do T' (VWAP +-2 Sigma)</option>
            <option value="composite">7. Composite Score</option>
            <option value="scale_ladder">8. Momentum Scale-Out Ladder</option>
            <option value="plain_tpsl">9. Plain TP/SL & Manual Sandbox</option>
        </select>
        <label class="chk-item" style="margin-top:8px; color:#ffd600; font-weight:600;">
            <input type="checkbox" id="enable_trailing_only" onchange="loadChartData()"> Rule: No Take-Profit / Trailing-Stop Only (Hold Until Trend Flip)
        </label>
        <button class="btn" onclick="loadChartData()">Apply Strategy & Rules</button>
    </details>
</div>

<div class="main">
    <div class="topbar">
        <div style="font-weight:700; color:#00b0ff;">Quant Lab: Trailing Stop / No-TP Rule Engine</div>
        <div class="tf-selector">
            <button class="tf-btn active" onclick="switchTF('1Y')">1Y</button>
            <button class="tf-btn" onclick="switchTF('1M')">1M</button>
            <button class="tf-btn" onclick="switchTF('1W')">1W</button>
            <button class="tf-btn" onclick="switchTF('1d')">1D</button>
            <button class="tf-btn" onclick="switchTF('1h')">1H</button>
            <button class="tf-btn" onclick="switchTF('15m')">15m</button>
            <button class="tf-btn" onclick="switchTF('5m')">5m</button>
            <button class="tf-btn" onclick="switchTF('1m')">1m</button>
        </div>
    </div>

    <div class="replay-bar">
        <div style="display:flex; align-items:center; gap:6px;">
            <span style="font-weight:700; color:#00b0ff; font-size:11.5px;">🎬 REPLAY (Backspace: -1 Bar | Enter: +1 Bar):</span>
            <button class="rp-btn" id="btn-replay-start" onclick="startBarReplay()">⏮ Start Replay</button>
            <button class="rp-btn" onclick="stepReplay(-1)">◀ -1 Bar</button>
            <button class="rp-btn" onclick="stepReplay(1)">▶ +1 Bar</button>
            <button class="rp-btn" onclick="stopBarReplay()">⏹ Exit</button>
            <span id="replay-status-text" style="color:var(--gold); font-weight:600;">[Full History]</span>
        </div>
    </div>

    <div class="alert-banner" id="alert-banner"></div>

    <div class="chart-wrapper" id="chart-wrap">
        <div id="tv-chart"></div>
    </div>

    <div class="bottom-panels">
        <details class="fold-card" open>
            <summary>Trade Execution Log</summary>
            <div style="overflow-x:auto; max-height:180px;">
                <table id="trade-log-table">
                    <thead>
                        <tr><th>#</th><th>Side</th><th>Signal</th><th>Entry Time</th><th>Entry $</th><th>Shares</th><th>SL / TP</th><th>Exit Time</th><th>Exit $</th><th>Result</th><th>PnL ($)</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </details>
    </div>
</div>

<script>
let chart, candleSeries, volumeSeries;
let currentData = null;
let currentTF = "1Y";
let replayActive = false, replayIndex = 0;
let asyncStart = null, asyncInterval = null;

function showLoading(msg) {
    asyncStart = Date.now();
    const b = document.getElementById('alert-banner');
    b.style.display = 'block';
    if (asyncInterval) clearInterval(asyncInterval);
    asyncInterval = setInterval(() => {
        b.textContent = `⏳ ${msg}... Elapsed: ${((Date.now() - asyncStart)/1000).toFixed(1)}s`;
    }, 100);
}
function hideLoading() {
    if (asyncInterval) clearInterval(asyncInterval);
    document.getElementById('alert-banner').style.display = 'none';
}

window.addEventListener('keydown', (e) => {
    if (!replayActive) return;
    if (e.key === 'Backspace') { e.preventDefault(); stepReplay(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); stepReplay(1); }
});

function initChart() {
    chart = LightweightCharts.createChart(document.getElementById('tv-chart'), {
        layout: { background: { type: 'solid', color: '#0b0e14' }, textColor: '#d1d4dc' },
        grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } }
    });
    candleSeries = chart.addCandlestickSeries({ upColor: '#00e676', downColor: '#ff1744' });
    volumeSeries = chart.addHistogramSeries({ priceScaleId: '' });
    window.addEventListener('resize', () => chart.applyOptions({ width: document.getElementById('chart-wrap').clientWidth, height: document.getElementById('chart-wrap').clientHeight }));
}

function updateStrategyGuide() {
    document.getElementById('strategy-guide-content').innerHTML = `
        <div><b>Strategies 1-9:</b> Standard trend, pullback, SMC, and golden pocket setups.</div>
        <div style="margin-top:4px; color:#ffd600;"><b>Trailing Stop Rule:</b> Removes fixed take-profit targets, continuously ratcheting stop-loss until trend reverses.</div>
    `;
}

function startBarReplay() {
    if (!currentData || !currentData.candles) return;
    replayActive = true;
    replayIndex = Math.max(1, Math.floor(currentData.candles.length * 0.65));
    document.getElementById('btn-replay-start').classList.add('active');
    renderReplay();
}

function stepReplay(delta) {
    if (!currentData || !currentData.candles) return;
    if (!replayActive) { startBarReplay(); return; }
    replayIndex = Math.max(0, Math.min(currentData.candles.length - 1, replayIndex + delta));
    renderReplay();
}

function stopBarReplay() {
    replayActive = false;
    document.getElementById('btn-replay-start').classList.remove('active');
    document.getElementById('replay-status-text').textContent = '[Full History]';
    if (currentData) { replayIndex = currentData.candles.length - 1; renderReplay(); }
}

function renderReplay() {
    if (!currentData) return;
    const maxIdx = replayActive ? replayIndex : (currentData.candles.length - 1);
    candleSeries.setData(currentData.candles.slice(0, maxIdx + 1));
    volumeSeries.setData(currentData.volumes.slice(0, maxIdx + 1));
    document.getElementById('replay-status-text').textContent = replayActive ? `Bar ${maxIdx + 1}/${currentData.candles.length}` : '[Full History]';
}

function switchTF(tf) {
    currentTF = tf;
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.toggle('active', b.textContent === tf));
    loadChartData();
}

async function loadChartData() {
    showLoading(`Loading ${currentTF} data for ${document.getElementById('ticker').value}`);
    const params = new URLSearchParams({
        ticker: document.getElementById('ticker').value, tf: currentTF,
        strategy: document.getElementById('strategy').value, trade_side: document.getElementById('trade_side').value,
        enable_trailing_only: document.getElementById('enable_trailing_only').checked ? 'true' : 'false'
    });
    const res = await fetch(`/api/analyze?${params.toString()}`);
    const data = await res.json();
    hideLoading();

    if (data.error) { alert(data.error); return; }
    currentData = data;
    updateStrategyGuide();
    if (!replayActive) replayIndex = data.candles.length - 1;
    renderReplay();
    updateTable(data);
}

function updateTable(data) {
    const tbody = document.querySelector('#trade-log-table tbody');
    tbody.innerHTML = '';
    (data.backtest.trades || []).forEach(t => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${t.trade_no}</td><td class="${t.side==='LONG'?'pos':'neg'}"><b>${t.side}</b></td><td>${t.signal_type}</td><td>${t.entry_dt}</td><td>$${t.entry_price}</td><td>${t.shares}</td><td>${t.sl_price} / ${t.tp_price}</td><td>${t.exit_dt}</td><td>$${t.exit_price}</td><td>${t.type}</td><td class="${t.pnl>=0?'pos':'neg'}">$${t.pnl}</td>`;
        tbody.appendChild(tr);
    });
}

window.onload = () => { initChart(); updateStrategyGuide(); loadChartData(); };
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
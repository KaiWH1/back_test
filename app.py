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

app = FastAPI(title="Quant Lab: Dual-System Re-Entry & Trade Inspector")

# ==============================================================================
# 0. STRICT JSON SANITIZER (PREVENTS NaN / Inf ValueError CRASHES)
# ==============================================================================
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

# ==============================================================================
# 1. DISK + MEMORY CACHE & MAX-LOOKBACK MULTI-SOURCE DATA ENGINE
# ==============================================================================
DATA_CACHE: Dict[str, pd.DataFrame] = {}
DATA_CACHE_TS: Dict[str, float] = {}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TIMEFRAME_HIERARCHY = ["1Y", "1M", "1W", "1d", "1h", "15m", "5m", "1m"]
NEXT_TIMEFRAME = {
    "1Y": "1M", "1M": "1W", "1W": "1d", "1d": "1h",
    "1h": "15m", "15m": "5m", "5m": "1m", "1m": "1m"
}
YF_LIMIT_DAYS = {
    "1h": 729,
    "15m": 59,
    "5m": 59,
    "1m": 28
}

def _direct_yahoo_v8_chart(ticker: str, interval: str, days_back: Optional[int] = None) -> pd.DataFrame:
    now_ts = int((datetime.now() + timedelta(days=2)).timestamp())
    start_ts = 0 if days_back is None else int((datetime.now() - timedelta(days=days_back)).timestamp())

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_ts,
        "period2": now_ts,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits"
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
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
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"]
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
    df = df[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan).dropna()
    return df

def _fetch_nasdaq_daily(ticker: str) -> pd.DataFrame:
    clean_t = ticker.upper().strip()
    end_dt = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    start_dt = "1970-01-01"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    for asset_class in ["stocks", "etf"]:
        url = (
            f"https://api.nasdaq.com/api/quote/{clean_t}/historical"
            f"?assetclass={asset_class}&fromdate={start_dt}&limit=99999&todate={end_dt}"
        )
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
            path[max_i] = h
            path[min_i] = l
            path[0] = o
            path[-1] = c
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
            all_rows.append({
                "dt": bar_t, "open": b_open, "high": b_high,
                "low": b_low, "close": b_close, "volume": b_vol
            })
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
        return df_syn, (
            f"Yahoo IP rate-limit active for '{interval}': Reconstructed max {max_days}d of {interval} bars "
            f"anchored to real Daily OHLCV."
        )
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
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
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

# ==============================================================================
# 2. QUANTITATIVE INDICATORS (WITH EMA 200) & DYNAMIC SL/TP FORMULA ENGINE
# ==============================================================================
def compute_all_indicators(
    df: pd.DataFrame,
    tf: str,
    atr_mult: float = 1.5,
    gk_mult: float = 3.0,
    dynamic_tpsl: bool = True,
    w_trend: float = 0.4,
    w_liq: float = 0.4,
    w_rev: float = 0.2
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

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    f_bull = df['fvg_bull'].values
    f_bear = df['fvg_bear'].values

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
        dyn_tp_raw = (
            gk_mult * (0.75 + 0.35 * trend_strength + 0.20 * rvol_boost) * (1.0 + 0.25 * df['composite_score'].abs())
        )
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

# ==============================================================================
# 3. DUAL-POOL PORTFOLIO, EMA 200 FILTER, CONTINUOUS RE-ENTRY & BACKTEST ENGINE
# ==============================================================================
STRATEGY_METADATA = {
    "trend": {
        "name": "1. Core Trend (EMA 20/50/200 + Volume)",
        "indicators": ["ema20", "ema50", "ema200", "atr_bands"],
        "bucket": "Core 70%",
        "guide": "Rides established macro trends. Enters on pullbacks toward EMA 20 and continuously re-enters on subsequent dips as long as the trend remains bullish."
    },
    "watch_dip": {
        "name": "2. Watch -> Pullback/Rally -> Buy/Short",
        "indicators": ["ema20", "ema50", "ema200", "vwap", "gk_bands"],
        "bucket": "Core 70%",
        "guide": "Monitors trend momentum, waits for measured pullback/rally exhaustion, and re-enters iteratively on repeated dips."
    },
    "smc_fvg": {
        "name": "3. SMC & Fair Value Gaps (Bull/Bear FVG)",
        "indicators": ["fvg_zones", "ema50", "ema200", "atr_bands"],
        "bucket": "Satellite 30%",
        "guide": "Smart Money Concepts: Trades repeated institutional Fair Value Gap retests during prevailing market trends."
    },
    "fib_golden": {
        "name": "4. Fibonacci Golden Pocket (0.618 / 0.382)",
        "indicators": ["fib_levels", "ema50", "ema200"],
        "bucket": "Core 70%",
        "guide": "Repeatedly buys the 0.618 golden pocket retracement in bull regimes or shorting the 0.382 zone in bear regimes."
    },
    "orb": {
        "name": "5. Opening Range Breakout/Breakdown (ORB)",
        "indicators": ["orb_levels", "vwap", "ema200"],
        "bucket": "Satellite 30%",
        "guide": "Intraday momentum opening range breakout capturing strong early session breakouts."
    },
    "do_t": {
        "name": "6. Intraday 'Do T' (VWAP +-2 Sigma L/S)",
        "indicators": ["vwap", "vwap_bands", "gk_bands", "ema200"],
        "bucket": "Satellite 30%",
        "guide": "Mean-reversion strategy fading extreme intraday extensions beyond VWAP bands."
    },
    "composite": {
        "name": "7. Composite Score (Alpha*A + Beta*B + Gamma*C)",
        "indicators": ["ema20", "ema200", "vwap", "gk_bands"],
        "bucket": "Core 70%",
        "guide": "Multi-factor quantitative alpha model combining trend slope, price pressure liquidity, and VWAP z-scores."
    },
    "scale_ladder": {
        "name": "8. Momentum Scale-Out Ladder (Sell xx% @ +xx% Tiers)",
        "indicators": ["ema20", "ema200", "vwap", "gk_bands", "atr_bands"],
        "bucket": "Core 70%",
        "guide": "Trend following with multi-tier partial profit taking and automatic trend re-entry on new dips."
    },
    "plain_tpsl": {
        "name": "9. Plain TP/SL & Manual Replay Sandbox",
        "indicators": ["ema20", "ema200", "atr_bands", "gk_bands"],
        "bucket": "Sandbox",
        "guide": "Manual sandbox mode for testing custom entries or fixed risk/reward parameters."
    },
    "no_tp_trend": {
        "name": "10. Dynamic Trailing SL / No Static TP (Trend Hold)",
        "indicators": ["ema20", "ema50", "ema200", "atr_bands"],
        "bucket": "Core 70%",
        "guide": "Never sets a static Take-Profit. Holds positions as long as trend is intact, continuously ratcheting trailing stop-loss behind price action."
    }
}

def _fmt_dt(dt_obj: pd.Timestamp, tf: str) -> str:
    if tf in ["1h", "15m", "5m", "1m"]:
        return dt_obj.strftime("%Y-%m-%d %H:%M")
    return dt_obj.strftime("%Y-%m-%d")

def run_strategy_backtest(
    df: pd.DataFrame,
    tf: str,
    strategy_key: str,
    trade_side: str = "both",
    ema200_filter: str = "none",
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    vol_model: str = "atr",
    enable_scale_out: bool = True,
    t1_gain_pct: float = 4.0,
    t1_sell_pct: float = 35.0,
    t2_gain_pct: float = 8.0,
    t2_sell_pct: float = 35.0,
    initial_capital: float = 100000.0,
    core_pct: float = 0.70,
    intraday_pct: float = 0.30,
    risk_per_trade_pct: float = 0.01,
    manual_entry_ts: Optional[int] = None,
    manual_side: str = "long",
    manual_alloc_pct: Optional[float] = None,
    use_original_engine: bool = False
) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"trades": [], "markers": [], "boxes": [], "stats": {}, "equity": []}

    allow_long = trade_side in ["both", "long"]
    allow_short = trade_side in ["both", "short"]
    is_no_tp = (strategy_key == "no_tp_trend")

    dt_index = df.index
    timestamps = [int(ts.timestamp()) for ts in dt_index]
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    atrs = df['atr'].values
    gks = df['gk_sigma_dollar'].values
    dyn_sl_arr = df['dyn_sl_mult'].values
    dyn_tp_arr = df['dyn_tp_mult'].values

    ema20 = df['ema20'].values
    ema50 = df['ema50'].values
    ema200 = df['ema200'].values
    rvol = df['rvol'].values
    vwap = df['vwap'].values
    vwap_l2 = df['vwap_l2'].values
    vwap_u2 = df['vwap_u2'].values
    fib_382 = df['fib_382'].values
    fib_618 = df['fib_618'].values
    fib_786 = df['fib_786'].values
    swing_high = df['swing_high'].values
    orb_high = df['orb_high'].values
    orb_low = df['orb_low'].values
    orb_mid = df['orb_mid'].values
    smc_bull = df['smc_fvg_bull_entry'].values
    smc_bear = df['smc_fvg_bear_entry'].values
    comp_score = df['composite_score'].values
    rsi = df['rsi'].values

    is_intraday_strat = strategy_key in ["orb", "do_t", "smc_fvg"]
    core_cash = initial_capital * core_pct
    intraday_cash = initial_capital * intraday_pct
    pool_label = f"Intraday ({int(intraday_pct*100)}%)" if is_intraday_strat else f"Core ({int(core_pct*100)}%)"

    peak_equity = initial_capital
    max_dd_pct = 0.0

    state = 0
    watch_peak = 0.0
    watch_trough = 0.0
    entry_price = 0.0
    sl_price = 0.0
    initial_sl_price = 0.0
    tp_price = 0.0
    alt_sl_price = 0.0
    alt_tp_price = 0.0
    entry_sl_mult_used = sl_mult
    entry_tp_mult_used = tp_mult
    entry_idx = 0
    initial_shares = 0
    remaining_shares = 0
    reserved_cash = 0.0
    realized_trade_pnl = 0.0
    scale_stage = 0
    t1_target_price = 0.0
    t2_target_price = 0.0
    scale_notes: List[str] = []
    entry_signal_reason = ""
    dollar_risk_at_entry = 0.0
    running_extreme = 0.0

    trades = []
    markers = []
    boxes = []
    equity_curve = []

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

        cur_sl_mult = sl_mult if use_original_engine else _clean_float(dyn_sl_arr[i], sl_mult, 2)
        cur_tp_mult = 999.0 if is_no_tp else (tp_mult if use_original_engine else _clean_float(dyn_tp_arr[i], tp_mult, 2))

        trend_long, trend_short = False, False
        trig_long, trig_short = False, False

        eff_strat = "trend" if is_no_tp else strategy_key

        if eff_strat == "trend":
            trend_long = c_close > ema50[i] and ema20[i] > ema50[i]
            trig_long = trend_long and (c_low <= ema20[i] * 1.008) and (c_close > c_open) and (rvol[i] >= 0.9)
            trend_short = c_close < ema50[i] and ema20[i] < ema50[i]
            trig_short = trend_short and (c_high >= ema20[i] * 0.992) and (c_close < c_open) and (rvol[i] >= 0.9)
        elif eff_strat == "watch_dip":
            trend_long = ema20[i] > ema50[i]
            trend_short = ema20[i] < ema50[i]
            if state in [0, 1, 3]:
                watch_peak = max(watch_peak, c_high)
                trig_long = ((watch_peak - c_low) >= 0.8 * unit_vol) and (c_close > c_open) and trend_long
            if state in [0, -1, -3]:
                watch_trough = min(watch_trough, c_low) if watch_trough > 0 else c_low
                trig_short = ((c_high - watch_trough) >= 0.8 * unit_vol) and (c_close < c_open) and trend_short
        elif eff_strat == "smc_fvg":
            trend_long = c_close >= ema50[i] * 0.98
            trig_long = bool(smc_bull[i]) and trend_long
            trend_short = c_close <= ema50[i] * 1.02
            trig_short = bool(smc_bear[i]) and trend_short
        elif eff_strat == "fib_golden":
            trend_long = c_close > fib_786[i] and ema20[i] >= ema50[i] * 0.99
            trig_long = trend_long and (c_low <= fib_618[i] * 1.005) and (c_close >= fib_618[i]) and (rsi[i] < 60)
            trend_short = c_close < swing_high[i] and ema20[i] < ema50[i] * 1.01
            trig_short = trend_short and (c_high >= fib_382[i] * 0.995) and (c_close <= fib_382[i]) and (rsi[i] > 40)
        elif eff_strat == "orb":
            if not np.isnan(orb_high[i]) and not np.isnan(orb_low[i]):
                trend_long = c_close > orb_mid[i] and c_close > vwap[i]
                trig_long = (closes[i - 1] <= orb_high[i - 1]) and (c_close > orb_high[i]) and (rvol[i] >= 1.2)
                trend_short = c_close < orb_mid[i] and c_close < vwap[i]
                trig_short = (closes[i - 1] >= orb_low[i - 1]) and (c_close < orb_low[i]) and (rvol[i] >= 1.2)
        elif eff_strat == "do_t":
            trend_long = c_close > ema50[i] * 0.93
            trig_long = (c_low <= vwap_l2[i]) and (c_close > c_open)
            trend_short = c_close < ema50[i] * 1.07
            trig_short = (c_high >= vwap_u2[i]) and (c_close < c_open)
        elif eff_strat == "composite":
            trend_long = comp_score[i] > 0.15
            trig_long = comp_score[i] >= 0.48
            trend_short = comp_score[i] < -0.15
            trig_short = comp_score[i] <= -0.48
        elif eff_strat == "scale_ladder":
            trend_long = (c_close > ema20[i]) and (ema20[i] >= ema50[i] * 0.995)
            trig_long = trend_long and (c_close > vwap[i]) and (rvol[i] >= 1.05) and (c_close > c_open)
            trend_short = (c_close < ema20[i]) and (ema20[i] <= ema50[i] * 1.005)
            trig_short = trend_short and (c_close < vwap[i]) and (rvol[i] >= 1.05) and (c_close < c_open)
        elif eff_strat == "plain_tpsl":
            if manual_entry_ts is not None:
                is_hit_bar = (ts >= manual_entry_ts) and (timestamps[i - 1] < manual_entry_ts or i == 1)
                if manual_side == "short":
                    trend_short = True
                    trig_short = is_hit_bar
                else:
                    trend_long = True
                    trig_long = is_hit_bar
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

        # STATE MACHINE TRANSITIONS (Continuous Re-Entry Logic)
        if state == 0:
            if strategy_key == "plain_tpsl" and manual_entry_ts is not None and (trig_long or trig_short):
                is_long = trig_long
                entry_price = c_close
                entry_idx = i
                entry_sl_mult_used = cur_sl_mult
                entry_tp_mult_used = cur_tp_mult
                safe_atr = _clean_float(atrs[i], c_close * 0.015)
                safe_gk = _clean_float(gks[i], safe_atr)
                if is_long:
                    sl_price = entry_price - cur_sl_mult * safe_atr
                    tp_price = entry_price + cur_tp_mult * safe_atr
                    alt_sl_price = entry_price - cur_sl_mult * safe_gk
                    alt_tp_price = entry_price + cur_tp_mult * safe_gk
                    running_extreme = c_high
                else:
                    sl_price = entry_price + cur_sl_mult * safe_atr
                    tp_price = entry_price - cur_tp_mult * safe_atr
                    alt_sl_price = entry_price + cur_sl_mult * safe_gk
                    alt_tp_price = entry_price - cur_tp_mult * safe_gk
                    running_extreme = c_low

                sh, d_risk = _calc_shares(entry_price, sl_price, force_alloc_pct=manual_alloc_pct)
                if sh > 0:
                    state = 2 if is_long else -2
                    initial_shares = sh
                    remaining_shares = sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage = 0
                    scale_notes = []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, is_long)
                    entry_signal_reason = f"MANUAL ({cur_sl_mult}x)"
                    markers.append({
                        "time": ts, "position": "belowBar" if is_long else "aboveBar",
                        "color": "#00E676" if is_long else "#FF4081",
                        "shape": "arrowUp" if is_long else "arrowDown",
                        "text": f"MANUAL {'BUY' if is_long else 'SHORT'} ({sh} sh @ ${entry_price:.2f})"
                    })
            elif trig_long:
                entry_price = c_close
                entry_idx = i
                entry_sl_mult_used = cur_sl_mult
                entry_tp_mult_used = cur_tp_mult
                sl_price = entry_price - cur_sl_mult * unit_vol
                tp_price = entry_price + cur_tp_mult * unit_vol
                running_extreme = c_high

                sh, d_risk = _calc_shares(entry_price, sl_price)
                if sh > 0:
                    state = 2
                    initial_shares = sh
                    remaining_shares = sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage = 0
                    scale_notes = []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, True)
                    entry_signal_reason = f"LONG ({cur_sl_mult}x)"
                    markers.append({
                        "time": ts, "position": "belowBar", "color": "#00E676",
                        "shape": "arrowUp", "text": f"BUY {sh} sh @ ${entry_price:.2f}"
                    })
            elif trig_short:
                entry_price = c_close
                entry_idx = i
                entry_sl_mult_used = cur_sl_mult
                entry_tp_mult_used = cur_tp_mult
                sl_price = entry_price + cur_sl_mult * unit_vol
                tp_price = entry_price - cur_tp_mult * unit_vol
                running_extreme = c_low

                sh, d_risk = _calc_shares(entry_price, sl_price)
                if sh > 0:
                    state = -2
                    initial_shares = sh
                    remaining_shares = sh
                    initial_sl_price = sl_price
                    dollar_risk_at_entry = d_risk
                    reserved_cash = sh * entry_price
                    _set_active_pool_cash(_get_active_pool_cash() - reserved_cash)
                    realized_trade_pnl = 0.0
                    scale_stage = 0
                    scale_notes = []
                    t1_target_price, t2_target_price = _compute_tier_targets(entry_price, tp_price, False)
                    entry_signal_reason = f"SHORT ({cur_sl_mult}x)"
                    markers.append({
                        "time": ts, "position": "aboveBar", "color": "#FF1744",
                        "shape": "arrowDown", "text": f"SHORT {sh} sh @ ${entry_price:.2f}"
                    })

        # ACTIVE LONG POSITION MANAGEMENT
        elif state == 2:
            if not use_original_engine:
                if c_high > running_extreme:
                    running_extreme = c_high
                potential_trailing_sl = running_extreme - (cur_sl_mult * unit_vol)
                if potential_trailing_sl > sl_price:
                    sl_price = potential_trailing_sl

            trend_reversal = (c_close < ema50[i] and ema20[i] < ema50[i]) if is_no_tp else False
            do_ladder = enable_scale_out or (strategy_key == "scale_ladder")
            hit_sl = c_low <= sl_price

            if do_ladder and not is_no_tp and not use_original_engine and not hit_sl and remaining_shares >= 3:
                if scale_stage == 0 and c_high >= t1_target_price and t1_target_price < tp_price:
                    qty1 = max(1, int(round(initial_shares * (t1_sell_pct / 100.0))))
                    qty1 = min(qty1, remaining_shares - 1)
                    if qty1 > 0:
                        part_pnl = (t1_target_price - entry_price) * qty1
                        realized_trade_pnl += part_pnl
                        _set_active_pool_cash(_get_active_pool_cash() + (qty1 * entry_price) + part_pnl)
                        remaining_shares -= qty1
                        reserved_cash = remaining_shares * entry_price
                        scale_stage = 1
                        sl_price = max(sl_price, entry_price)
                        scale_notes.append(f"T1(-{qty1}sh@${t1_target_price:.2f})")
                        markers.append({
                            "time": ts, "position": "aboveBar", "color": "#80D8FF",
                            "shape": "circle", "text": f"T1 SELL {qty1}sh (+${part_pnl:,.0f})"
                        })

                if scale_stage == 1 and c_high >= t2_target_price and t2_target_price < tp_price and remaining_shares >= 2:
                    qty2 = max(1, int(round(initial_shares * (t2_sell_pct / 100.0))))
                    qty2 = min(qty2, remaining_shares - 1)
                    if qty2 > 0:
                        part_pnl = (t2_target_price - entry_price) * qty2
                        realized_trade_pnl += part_pnl
                        _set_active_pool_cash(_get_active_pool_cash() + (qty2 * entry_price) + part_pnl)
                        remaining_shares -= qty2
                        reserved_cash = remaining_shares * entry_price
                        scale_stage = 2
                        sl_price = max(sl_price, t1_target_price)
                        scale_notes.append(f"T2(-{qty2}sh@${t2_target_price:.2f})")
                        markers.append({
                            "time": ts, "position": "aboveBar", "color": "#40C4FF",
                            "shape": "circle", "text": f"T2 SELL {qty2}sh (+${part_pnl:,.0f})"
                        })

            hit_tp = (c_high >= tp_price) and not is_no_tp
            hit_sl = c_low <= sl_price
            exit_on_trend = is_no_tp and trend_reversal

            if hit_tp or hit_sl or exit_on_trend:
                if hit_tp:
                    exit_price = tp_price
                    res_type = "TP"
                elif hit_sl:
                    exit_price = sl_price
                    res_type = "SL"
                else:
                    exit_price = c_close
                    res_type = "TREND-EXIT"

                rem_pnl = (exit_price - entry_price) * remaining_shares
                _set_active_pool_cash(_get_active_pool_cash() + (remaining_shares * entry_price) + rem_pnl)
                total_pnl = realized_trade_pnl + rem_pnl
                pos_val = entry_price * initial_shares
                ret_pct = (total_pnl / max(pos_val, 1e-6)) * 100.0
                remaining_shares = 0
                reserved_cash = 0.0
                state = 0
                cur_eq = _total_portfolio_equity(c_close)
                alloc_pct = (pos_val / max(cur_eq - total_pnl, 1.0)) * 100.0
                ladder_str = " -> ".join(scale_notes) if scale_notes else "100% @ Exit"

                trades.append({
                    "trade_no": len(trades) + 1, "side": "LONG", "signal_type": entry_signal_reason,
                    "pool": pool_label, "entry_bar_idx": int(entry_idx), "exit_bar_idx": int(i),
                    "entry_time": timestamps[entry_idx], "entry_dt": _fmt_dt(dt_index[entry_idx], tf),
                    "exit_time": ts, "exit_dt": _fmt_dt(dt_index[i], tf), "type": res_type,
                    "entry_price": _clean_float(entry_price, 0.0, 2), "exit_price": _clean_float(exit_price, 0.0, 2),
                    "sl_price": _clean_float(initial_sl_price, 0.0, 2), "tp_price": _clean_float(tp_price, 0.0, 2) if not is_no_tp else 0.0,
                    "dyn_mults": f"{entry_sl_mult_used:.2f}x", "scale_summary": ladder_str,
                    "shares": int(initial_shares), "position_value": _clean_float(pos_val, 0.0, 2),
                    "alloc_pct": _clean_float(alloc_pct, 0.0, 1), "dollar_risk": _clean_float(dollar_risk_at_entry, 0.0, 2),
                    "bar_volume": int(_clean_float(volumes[entry_idx], 0.0)), "bar_rvol": _clean_float(rvol[entry_idx], 1.0, 2),
                    "pnl": _clean_float(total_pnl, 0.0, 2), "ret_pct": _clean_float(ret_pct, 0.0, 2),
                    "equity_after": _clean_float(cur_eq, initial_capital, 2)
                })
                boxes.append({
                    "trade_no": len(trades), "side": "LONG", "t1": timestamps[entry_idx], "t2": ts,
                    "entry": _clean_float(entry_price), "tp": _clean_float(tp_price) if not is_no_tp else _clean_float(entry_price),
                    "sl": _clean_float(initial_sl_price), "t1_lvl": _clean_float(t1_target_price),
                    "t2_lvl": _clean_float(t2_target_price), "alt_tp": _clean_float(alt_tp_price),
                    "alt_sl": _clean_float(alt_sl_price), "outcome": res_type
                })
                markers.append({
                    "time": ts, "position": "aboveBar",
                    "color": "#FFD600" if total_pnl > 0 else "#FF1744",
                    "shape": "arrowDown", "text": f"L-{res_type} ({'+' if total_pnl > 0 else ''}${total_pnl:,.0f})"
                })

        # ACTIVE SHORT POSITION MANAGEMENT
        elif state == -2:
            if not use_original_engine:
                if c_low < running_extreme:
                    running_extreme = c_low
                potential_trailing_sl = running_extreme + (cur_sl_mult * unit_vol)
                if potential_trailing_sl < sl_price:
                    sl_price = potential_trailing_sl

            trend_reversal = (c_close > ema50[i] and ema20[i] > ema50[i]) if is_no_tp else False
            do_ladder = enable_scale_out or (strategy_key == "scale_ladder")
            hit_sl = c_high >= sl_price

            if do_ladder and not is_no_tp and not use_original_engine and not hit_sl and remaining_shares >= 3:
                if scale_stage == 0 and c_low <= t1_target_price and t1_target_price > tp_price:
                    qty1 = max(1, int(round(initial_shares * (t1_sell_pct / 100.0))))
                    qty1 = min(qty1, remaining_shares - 1)
                    if qty1 > 0:
                        part_pnl = (entry_price - t1_target_price) * qty1
                        realized_trade_pnl += part_pnl
                        _set_active_pool_cash(_get_active_pool_cash() + (qty1 * entry_price) + part_pnl)
                        remaining_shares -= qty1
                        reserved_cash = remaining_shares * entry_price
                        scale_stage = 1
                        sl_price = min(sl_price, entry_price)
                        scale_notes.append(f"T1(-{qty1}sh@${t1_target_price:.2f})")
                        markers.append({
                            "time": ts, "position": "belowBar", "color": "#80D8FF",
                            "shape": "circle", "text": f"T1 COVER {qty1}sh (+${part_pnl:,.0f})"
                        })

                if scale_stage == 1 and c_low <= t2_target_price and t2_target_price > tp_price and remaining_shares >= 2:
                    qty2 = max(1, int(round(initial_shares * (t2_sell_pct / 100.0))))
                    qty2 = min(qty2, remaining_shares - 1)
                    if qty2 > 0:
                        part_pnl = (entry_price - t2_target_price) * qty2
                        realized_trade_pnl += part_pnl
                        _set_active_pool_cash(_get_active_pool_cash() + (qty2 * entry_price) + part_pnl)
                        remaining_shares -= qty2
                        reserved_cash = remaining_shares * entry_price
                        scale_stage = 2
                        sl_price = min(sl_price, t1_target_price)
                        scale_notes.append(f"T2(-{qty2}sh@${t2_target_price:.2f})")
                        markers.append({
                            "time": ts, "position": "belowBar", "color": "#40C4FF",
                            "shape": "circle", "text": f"T2 COVER {qty2}sh (+${part_pnl:,.0f})"
                        })

            hit_tp = (c_low <= tp_price) and not is_no_tp
            hit_sl = c_high >= sl_price
            exit_on_trend = is_no_tp and trend_reversal

            if hit_tp or hit_sl or exit_on_trend:
                if hit_tp:
                    exit_price = tp_price
                    res_type = "TP"
                elif hit_sl:
                    exit_price = sl_price
                    res_type = "SL"
                else:
                    exit_price = c_close
                    res_type = "TREND-EXIT"

                rem_pnl = (entry_price - exit_price) * remaining_shares
                _set_active_pool_cash(_get_active_pool_cash() + (remaining_shares * entry_price) + rem_pnl)
                total_pnl = realized_trade_pnl + rem_pnl
                pos_val = entry_price * initial_shares
                ret_pct = (total_pnl / max(pos_val, 1e-6)) * 100.0
                remaining_shares = 0
                reserved_cash = 0.0
                state = 0
                cur_eq = _total_portfolio_equity(c_close)
                alloc_pct = (pos_val / max(cur_eq - total_pnl, 1.0)) * 100.0
                ladder_str = " -> ".join(scale_notes) if scale_notes else "100% @ Exit"

                trades.append({
                    "trade_no": len(trades) + 1, "side": "SHORT", "signal_type": entry_signal_reason,
                    "pool": pool_label, "entry_bar_idx": int(entry_idx), "exit_bar_idx": int(i),
                    "entry_time": timestamps[entry_idx], "entry_dt": _fmt_dt(dt_index[entry_idx], tf),
                    "exit_time": ts, "exit_dt": _fmt_dt(dt_index[i], tf), "type": res_type,
                    "entry_price": _clean_float(entry_price, 0.0, 2), "exit_price": _clean_float(exit_price, 0.0, 2),
                    "sl_price": _clean_float(initial_sl_price, 0.0, 2), "tp_price": _clean_float(tp_price, 0.0, 2) if not is_no_tp else 0.0,
                    "dyn_mults": f"{entry_sl_mult_used:.2f}x", "scale_summary": ladder_str,
                    "shares": int(initial_shares), "position_value": _clean_float(pos_val, 0.0, 2),
                    "alloc_pct": _clean_float(alloc_pct, 0.0, 1), "dollar_risk": _clean_float(dollar_risk_at_entry, 0.0, 2),
                    "bar_volume": int(_clean_float(volumes[entry_idx], 0.0)), "bar_rvol": _clean_float(rvol[entry_idx], 1.0, 2),
                    "pnl": _clean_float(total_pnl, 0.0, 2), "ret_pct": _clean_float(ret_pct, 0.0, 2),
                    "equity_after": _clean_float(cur_eq, initial_capital, 2)
                })
                boxes.append({
                    "trade_no": len(trades), "side": "SHORT", "t1": timestamps[entry_idx], "t2": ts,
                    "entry": _clean_float(entry_price), "tp": _clean_float(tp_price) if not is_no_tp else _clean_float(entry_price),
                    "sl": _clean_float(initial_sl_price), "t1_lvl": _clean_float(t1_target_price),
                    "t2_lvl": _clean_float(t2_target_price), "alt_tp": _clean_float(alt_tp_price),
                    "alt_sl": _clean_float(alt_sl_price), "outcome": res_type
                })
                markers.append({
                    "time": ts, "position": "belowBar",
                    "color": "#FFD600" if total_pnl > 0 else "#FF1744",
                    "shape": "arrowUp", "text": f"S-{res_type} ({'+' if total_pnl > 0 else ''}${total_pnl:,.0f})"
                })

        live_eq = _total_portfolio_equity(c_close)
        peak_equity = max(peak_equity, live_eq)
        dd = ((peak_equity - live_eq) / max(peak_equity, 1.0)) * 100.0
        max_dd_pct = max(max_dd_pct, dd)
        equity_curve.append({"time": ts, "value": _clean_float(live_eq, initial_capital, 2)})

    if state in [2, -2] and n > 0 and remaining_shares > 0:
        is_long_open = (state == 2)
        last_c = closes[-1]
        rem_unreal = ((last_c - entry_price) if is_long_open else (entry_price - last_c)) * remaining_shares
        total_open_pnl = realized_trade_pnl + rem_unreal
        pos_val = entry_price * initial_shares
        unreal_pct = (total_open_pnl / max(pos_val, 1e-6)) * 100.0
        final_eq = _total_portfolio_equity(last_c)
        alloc_pct = (pos_val / max(final_eq, 1.0)) * 100.0
        ladder_str = (" -> ".join(scale_notes) + f" ({remaining_shares}sh open)") if scale_notes else f"OPEN ({remaining_shares}sh)"

        trades.append({
            "trade_no": len(trades) + 1, "side": "LONG" if is_long_open else "SHORT",
            "signal_type": entry_signal_reason, "pool": pool_label, "entry_bar_idx": int(entry_idx),
            "exit_bar_idx": int(n - 1), "entry_time": timestamps[entry_idx], "entry_dt": _fmt_dt(dt_index[entry_idx], tf),
            "exit_time": timestamps[-1], "exit_dt": "OPEN (Active)", "type": "OPEN",
            "entry_price": _clean_float(entry_price, 0.0, 2), "exit_price": _clean_float(last_c, 0.0, 2),
            "sl_price": _clean_float(initial_sl_price, 0.0, 2), "tp_price": _clean_float(tp_price, 0.0, 2),
            "dyn_mults": f"{entry_sl_mult_used:.2f}x", "scale_summary": ladder_str,
            "shares": int(initial_shares), "position_value": _clean_float(pos_val, 0.0, 2),
            "alloc_pct": _clean_float(alloc_pct, 0.0, 1), "dollar_risk": _clean_float(dollar_risk_at_entry, 0.0, 2),
            "bar_volume": int(_clean_float(volumes[entry_idx], 0.0)), "bar_rvol": _clean_float(rvol[entry_idx], 1.0, 2),
            "pnl": _clean_float(total_open_pnl, 0.0, 2), "ret_pct": _clean_float(unreal_pct, 0.0, 2),
            "equity_after": _clean_float(final_eq, initial_capital, 2)
        })
        boxes.append({
            "trade_no": len(trades), "side": "LONG" if is_long_open else "SHORT",
            "t1": timestamps[entry_idx], "t2": timestamps[-1], "entry": _clean_float(entry_price),
            "tp": _clean_float(tp_price), "sl": _clean_float(initial_sl_price),
            "t1_lvl": _clean_float(t1_target_price), "t2_lvl": _clean_float(t2_target_price),
            "alt_tp": _clean_float(alt_tp_price), "alt_sl": _clean_float(alt_sl_price), "outcome": "ACTIVE"
        })

    final_mark_equity = _total_portfolio_equity(closes[-1]) if n > 0 else initial_capital
    closed_trades = [t for t in trades if t["type"] in ["TP", "TRAIL-WIN", "SL", "TREND-EXIT", "MANUAL-WIN", "MANUAL-EXIT"]]
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
        "strategy_key": strategy_key,
        "strategy_name": STRATEGY_METADATA[strategy_key]["name"],
        "bucket": STRATEGY_METADATA[strategy_key]["bucket"],
        "initial_capital": _clean_float(initial_capital, 100000.0, 2),
        "ending_equity": _clean_float(final_mark_equity, initial_capital, 2),
        "core_cash_end": _clean_float(core_cash, initial_capital * core_pct, 2),
        "intraday_cash_end": _clean_float(intraday_cash, initial_capital * intraday_pct, 2),
        "net_pnl": _clean_float(final_mark_equity - initial_capital, 0.0, 2),
        "net_return_pct": _clean_float(net_return_pct, 0.0, 2),
        "buy_hold_pct": _clean_float(buy_hold_ret, 0.0, 2),
        "total_trades": total_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "win_rate_pct": _clean_float(win_rate, 0.0, 1),
        "profit_factor": _clean_float(min(profit_factor, 99.9), 0.0, 2),
        "max_dd_pct": _clean_float(max_dd_pct, 0.0, 2)
    }

    return {
        "trades": trades,
        "markers": markers,
        "boxes": boxes,
        "stats": stats,
        "equity": equity_curve
    }

# ==============================================================================
# 4. BLACK-SCHOLES GREEKS & YAHOO LIVE OPTION CHAIN (UP TO 60 DAYS) + HISTORY
# ==============================================================================
def black_scholes_analysis(
    S: float, K: float, days_to_expiry: int, iv_pct: float,
    r_pct: float, gk_daily_vol_pct: float, opt_type: str = "call"
) -> Dict[str, Any]:
    S = max(_clean_float(S, 100.0), 0.01)
    K = max(_clean_float(K, 100.0), 0.01)
    T = max(_clean_float(days_to_expiry, 14) / 365.0, 1 / 3650.0)
    sigma = max(_clean_float(iv_pct, 50.0) / 100.0, 0.01)
    r = _clean_float(r_pct, 4.25) / 100.0
    gk_daily = max(_clean_float(gk_daily_vol_pct, 2.0), 0.1)
    hv_annual = (gk_daily / 100.0) * math.sqrt(252)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if opt_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
        prob_itm = norm.cdf(d2) * 100.0
        breakeven = K + price
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
        prob_itm = norm.cdf(-d2) * 100.0
        breakeven = K - price

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100.0
    expected_move_iv_dollar = S * sigma * math.sqrt(T)
    expected_move_hv_dollar = S * hv_annual * math.sqrt(T)
    iv_hv_ratio = sigma / max(hv_annual, 0.01)

    if iv_hv_ratio > 1.25:
        verdict = "EXPENSIVE (IV Crush Warning: IV > Realized GK Vol)"
    elif iv_hv_ratio < 0.85:
        verdict = "CHEAP / UNDERPRICED (Positive Vega Edge: IV < Realized GK Vol)"
    else:
        verdict = "FAIRLY PRICED (IV aligned with Garman-Klass HV)"

    return _sanitize_for_json({
        "theo_price": _clean_float(price, 0.01, 3),
        "delta": _clean_float(delta, 0.0, 3),
        "gamma": _clean_float(gamma, 0.0, 4),
        "theta_daily": _clean_float(theta, 0.0, 3),
        "vega_per_1pct": _clean_float(vega, 0.0, 3),
        "prob_itm_pct": _clean_float(prob_itm, 0.0, 1),
        "breakeven": _clean_float(breakeven, S, 2),
        "hv_annual_pct": _clean_float(hv_annual * 100.0, 20.0, 1),
        "iv_hv_ratio": _clean_float(iv_hv_ratio, 1.0, 2),
        "expected_move_iv": _clean_float(expected_move_iv_dollar, 1.0, 2),
        "expected_move_hv": _clean_float(expected_move_hv_dollar, 1.0, 2),
        "verdict": verdict
    })

@app.get("/api/option_chain")
def api_option_chain(
    ticker: str = Query("SNDK"),
    expiry: Optional[str] = Query(None),
    spot: float = Query(100.0),
    gk_daily_vol_pct: float = Query(2.5),
    max_dte: int = Query(60)
):
    ticker = ticker.upper().strip()
    spot = max(_clean_float(spot, 100.0), 1.0)
    gk_daily_vol_pct = max(_clean_float(gk_daily_vol_pct, 2.5), 0.2)
    now_dt = datetime.now()
    refresh_stamp = now_dt.strftime("%H:%M:%S")

    try:
        tk = yf.Ticker(ticker)
        all_expirations = list(tk.options)
        if all_expirations:
            exp_up_to_60d = []
            for exp_str in all_expirations:
                try:
                    dte_val = (datetime.strptime(exp_str, "%Y-%m-%d") - now_dt).days
                    if 0 <= dte_val <= max_dte:
                        exp_up_to_60d.append(exp_str)
                except Exception:
                    continue
            expirations = exp_up_to_60d if exp_up_to_60d else all_expirations[:6]
            sel_exp = expiry if (expiry in expirations) else expirations[0]
            chain = tk.option_chain(sel_exp)
            dte = max(1, (datetime.strptime(sel_exp, "%Y-%m-%d") - now_dt).days)

            def format_side(df_side: pd.DataFrame, opt_type: str):
                df_sub = df_side.copy()
                df_sub["dist"] = (df_sub["strike"] - spot).abs()
                df_sub = df_sub.sort_values("dist").head(16).sort_values("strike")
                rows = []
                for _, r in df_sub.iterrows():
                    strike_v = _clean_float(r.get("strike"), spot, 2)
                    iv_val = _clean_float(r.get("impliedVolatility"), 0.5) * 100.0
                    bs = black_scholes_analysis(spot, strike_v, dte, max(iv_val, 5.0), 4.25, gk_daily_vol_pct, opt_type)
                    rows.append({
                        "contractSymbol": str(r.get("contractSymbol", "")),
                        "strike": strike_v,
                        "dte": dte,
                        "lastPrice": _clean_float(r.get("lastPrice"), bs["theo_price"], 2),
                        "bid": _clean_float(r.get("bid"), 0.0, 2),
                        "ask": _clean_float(r.get("ask"), 0.0, 2),
                        "iv_pct": _clean_float(iv_val, 50.0, 1),
                        "volume": int(_clean_float(r.get("volume"), 0)),
                        "openInterest": int(_clean_float(r.get("openInterest"), 0)),
                        "delta": bs["delta"],
                        "theta": bs["theta_daily"],
                        "verdict": "CHEAP" if bs["iv_hv_ratio"] < 0.9 else ("EXPENSIVE" if bs["iv_hv_ratio"] > 1.25 else "FAIR")
                    })
                return rows

            return _sanitize_for_json({
                "source": f"Yahoo Finance Live Option Chain (<= {max_dte}d | Updated {refresh_stamp})",
                "updated_at": refresh_stamp,
                "expirations": expirations,
                "selected_expiry": sel_exp,
                "dte": dte,
                "calls": format_side(chain.calls, "call"),
                "puts": format_side(chain.puts, "put")
            })
    except Exception:
        pass

    exp_dates = [(now_dt + timedelta(days=d)).strftime("%Y-%m-%d") for d in [7, 14, 21, 30, 45, 60]]
    sel_exp = expiry if (expiry in exp_dates) else exp_dates[1]
    dte = max(1, (datetime.strptime(sel_exp, "%Y-%m-%d") - now_dt).days)
    hv_ann = max((gk_daily_vol_pct / 100.0) * math.sqrt(252) * 100.0, 25.0)
    step = max(1.0, round(spot * 0.025, 0))
    base_strike = round(spot / step) * step
    calls, puts = [], []
    for k_idx in range(-6, 7):
        k = round(base_strike + k_idx * step, 2)
        if k <= 0:
            continue
        skew_iv = hv_ann * (1.05 + 0.02 * abs(k_idx))
        bs_c = black_scholes_analysis(spot, k, dte, skew_iv, 4.25, gk_daily_vol_pct, "call")
        bs_p = black_scholes_analysis(spot, k, dte, skew_iv, 4.25, gk_daily_vol_pct, "put")
        ymd = sel_exp.replace("-", "")[2:]
        c_sym = f"{ticker}{ymd}C{int(k*1000):08d}"
        p_sym = f"{ticker}{ymd}P{int(k*1000):08d}"
        calls.append({
            "contractSymbol": c_sym, "strike": k, "dte": dte,
            "lastPrice": _clean_float(bs_c["theo_price"], 0.01, 2),
            "bid": _clean_float(max(0.01, bs_c["theo_price"] * 0.96), 0.01, 2),
            "ask": _clean_float(bs_c["theo_price"] * 1.04, 0.02, 2),
            "iv_pct": _clean_float(skew_iv, 50.0, 1),
            "volume": 120 * (7 - abs(k_idx)), "openInterest": 850 * (7 - abs(k_idx)),
            "delta": bs_c["delta"], "theta": bs_c["theta_daily"],
            "verdict": "CHEAP" if bs_c["iv_hv_ratio"] < 0.9 else ("EXPENSIVE" if bs_c["iv_hv_ratio"] > 1.25 else "FAIR")
        })
        puts.append({
            "contractSymbol": p_sym, "strike": k, "dte": dte,
            "lastPrice": _clean_float(bs_p["theo_price"], 0.01, 2),
            "bid": _clean_float(max(0.01, bs_p["theo_price"] * 0.96), 0.01, 2),
            "ask": _clean_float(bs_p["theo_price"] * 1.04, 0.02, 2),
            "iv_pct": _clean_float(skew_iv, 50.0, 1),
            "volume": 95 * (7 - abs(k_idx)), "openInterest": 720 * (7 - abs(k_idx)),
            "delta": bs_p["delta"], "theta": bs_p["theta_daily"],
            "verdict": "CHEAP" if bs_p["iv_hv_ratio"] < 0.9 else ("EXPENSIVE" if bs_p["iv_hv_ratio"] > 1.25 else "FAIR")
        })

    return _sanitize_for_json({
        "source": f"BSM + Garman-Klass Live Chain (<= 60d | Updated {refresh_stamp})",
        "updated_at": refresh_stamp,
        "expirations": exp_dates,
        "selected_expiry": sel_exp,
        "dte": dte,
        "calls": calls,
        "puts": puts
    })

@app.get("/api/option_contract_history")
def api_option_contract_history(
    contract_symbol: str = Query(...),
    ticker: str = Query("SNDK"),
    strike: float = Query(100.0),
    opt_type: str = Query("call"),
    iv_pct: float = Query(55.0)
):
    refresh_stamp = datetime.now().strftime("%H:%M:%S")
    try:
        opt_tk = yf.Ticker(contract_symbol)
        df_opt = opt_tk.history(period="60d", interval="1d")
        if df_opt.empty:
            df_opt = opt_tk.history(period="max", interval="1d")
        if not df_opt.empty:
            df_opt.index = pd.to_datetime(df_opt.index).tz_localize(None)
            bars = [
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "close": _clean_float(r["Close"], 0.01, 2),
                    "high": _clean_float(r["High"], 0.01, 2),
                    "low": _clean_float(r["Low"], 0.01, 2),
                    "volume": int(_clean_float(r.get("Volume", 0), 0))
                }
                for dt, r in df_opt.tail(60).iterrows()
            ]
            return _sanitize_for_json({
                "contract": contract_symbol,
                "source": f"Yahoo Finance 60-Day Contract History (Updated {refresh_stamp})",
                "bars": bars
            })
    except Exception:
        pass

    df_stock, _ = fetch_and_construct_tf(ticker, "1d")
    df_sub = df_stock.tail(60)
    bars = []
    total_days = len(df_sub)
    for idx, (dt, r) in enumerate(df_sub.iterrows()):
        rem_dte = max(7, total_days - idx + 14)
        bs = black_scholes_analysis(float(r["close"]), strike, rem_dte, iv_pct, 4.25, 2.5, opt_type)
        bars.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": _clean_float(bs["theo_price"], 0.01, 2),
            "high": _clean_float(bs["theo_price"] * 1.08, 0.01, 2),
            "low": _clean_float(max(0.01, bs["theo_price"] * 0.92), 0.01, 2),
            "volume": int(max(10, _clean_float(r["volume"], 1000) / 5000))
        })
    return _sanitize_for_json({
        "contract": contract_symbol,
        "source": f"60-Day Black-Scholes Reconstructed History (Updated {refresh_stamp})",
        "bars": bars
    })

# ==============================================================================
# 5. MAIN BACKTEST, BAR INSPECTOR & CHART FASTAPI ENDPOINT
# ==============================================================================
@app.get("/api/analyze")
def api_analyze(
    ticker: str = Query("SNDK"),
    tf: str = Query("1Y"),
    start_ts: Optional[int] = Query(None),
    cutoff_ts: Optional[int] = Query(None),
    strategy: str = Query("trend"),
    trade_side: str = Query("both"),
    ema200_filter: str = Query("none"),
    sl_mult: float = Query(1.5),
    tp_mult: float = Query(3.0),
    dynamic_tpsl: bool = Query(True),
    vol_model: str = Query("atr"),
    enable_scale_out: bool = Query(True),
    t1_gain_pct: float = Query(4.0),
    t1_sell_pct: float = Query(35.0),
    t2_gain_pct: float = Query(8.0),
    t2_sell_pct: float = Query(35.0),
    w_trend: float = Query(0.4),
    w_liq: float = Query(0.4),
    w_rev: float = Query(0.2),
    initial_capital: float = Query(100000.0),
    core_pct: float = Query(70.0),
    intraday_pct: float = Query(30.0),
    risk_pct: float = Query(1.0),
    manual_entry_ts: Optional[int] = Query(None),
    manual_side: str = Query("long"),
    manual_alloc_pct: Optional[float] = Query(None),
    engine_mode: str = Query("dynamic"),
    force_refresh: bool = Query(False)
):
    use_original = (engine_mode == "original")
    df_raw, warn_msg = fetch_and_construct_tf(ticker, tf, force_refresh=force_refresh)
    if df_raw.empty:
        return JSONResponse({"error": warn_msg or "No data available."}, status_code=400)

    first_avail_ts = int(df_raw.index[0].timestamp())
    if start_ts is not None and start_ts < first_avail_ts:
        first_dt_str = df_raw.index[0].strftime("%Y-%m-%d %H:%M")
        extra_warn = (
            f"Max Yahoo lookback limit: '{tf}' data is unavailable prior to {first_dt_str}. "
            f"Showing all available '{tf}' bars starting from {first_dt_str}."
        )
        warn_msg = f"{warn_msg} | {extra_warn}" if warn_msg else extra_warn
        start_ts = first_avail_ts

    if cutoff_ts is not None:
        cutoff_dt = datetime.fromtimestamp(cutoff_ts)
        df_raw = df_raw[df_raw.index <= cutoff_dt]

    if start_ts is not None:
        start_dt = datetime.fromtimestamp(start_ts)
        mask_after = df_raw.index >= start_dt
        if mask_after.any():
            first_idx = int(np.argmax(mask_after))
            warmup_start_idx = max(0, first_idx - 220)
            df_calc = df_raw.iloc[warmup_start_idx:].copy()
        else:
            df_calc = df_raw.copy()
    else:
        df_calc = df_raw.copy()

    df_ind, unavailable_list = compute_all_indicators(
        df_calc, tf, atr_mult=sl_mult, gk_mult=tp_mult,
        dynamic_tpsl=dynamic_tpsl, w_trend=w_trend, w_liq=w_liq, w_rev=w_rev
    )

    if start_ts is not None:
        start_dt = datetime.fromtimestamp(start_ts)
        df_view = df_ind[df_ind.index >= start_dt].copy()
        if df_view.empty:
            df_view = df_ind.copy()
    else:
        df_view = df_ind.copy()

    bt_selected = run_strategy_backtest(
        df_view, tf=tf, strategy_key=strategy, trade_side=trade_side,
        ema200_filter=ema200_filter, sl_mult=sl_mult, tp_mult=tp_mult,
        vol_model=vol_model, enable_scale_out=enable_scale_out,
        t1_gain_pct=t1_gain_pct, t1_sell_pct=t1_sell_pct,
        t2_gain_pct=t2_gain_pct, t2_sell_pct=t2_sell_pct,
        initial_capital=initial_capital, core_pct=core_pct / 100.0,
        intraday_pct=intraday_pct / 100.0, risk_per_trade_pct=risk_pct / 100.0,
        manual_entry_ts=manual_entry_ts, manual_side=manual_side, manual_alloc_pct=manual_alloc_pct,
        use_original_engine=use_original
    )

    all_summary = []
    for s_key in STRATEGY_METADATA.keys():
        res = run_strategy_backtest(
            df_view, tf=tf, strategy_key=s_key, trade_side=trade_side,
            ema200_filter=ema200_filter, sl_mult=sl_mult, tp_mult=tp_mult,
            vol_model=vol_model, enable_scale_out=enable_scale_out,
            t1_gain_pct=t1_gain_pct, t1_sell_pct=t1_sell_pct,
            t2_gain_pct=t2_gain_pct, t2_sell_pct=t2_sell_pct,
            initial_capital=initial_capital, core_pct=core_pct / 100.0,
            intraday_pct=intraday_pct / 100.0, risk_per_trade_pct=risk_pct / 100.0,
            manual_entry_ts=manual_entry_ts if s_key == "plain_tpsl" else None,
            manual_side=manual_side,
            manual_alloc_pct=manual_alloc_pct if s_key == "plain_tpsl" else None,
            use_original_engine=use_original
        )
        all_summary.append(res["stats"])

    candles = []
    volumes = []
    bar_details = {}
    series_data = {
        "ema20": [], "ema50": [], "ema200": [], "vwap": [],
        "vwap_u2": [], "vwap_l2": [], "atr_upper": [], "atr_lower": [],
        "gk_upper": [], "gk_lower": [], "fib_382": [], "fib_500": [],
        "fib_618": [], "fib_786": [], "orb_high": [], "orb_low": []
    }
    fvg_boxes = []
    unclear_spans = []
    in_unclear = False
    unclear_start = None

    idx_timestamps = [int(t.timestamp()) for t in df_view.index]
    is_intraday_strat = strategy in ["orb", "do_t", "smc_fvg"]
    pool_frac = (intraday_pct / 100.0) if is_intraday_strat else (core_pct / 100.0)
    risk_frac = risk_pct / 100.0

    for i, (dt, row) in enumerate(df_view.iterrows()):
        ts = idx_timestamps[i]
        o_val = _clean_float(row['open'], 0.0, 4)
        h_val = _clean_float(row['high'], 0.0, 4)
        l_val = _clean_float(row['low'], 0.0, 4)
        c_val = _clean_float(row['close'], 0.0, 4)
        v_val = _clean_float(row['volume'], 0.0)
        atr_val = _clean_float(row['atr'], max(c_val * 0.015, 0.01))
        gk_val = _clean_float(row['gk_sigma_dollar'], atr_val)
        d_sl = sl_mult if use_original else _clean_float(row['dyn_sl_mult'], sl_mult, 2)
        d_tp = tp_mult if use_original else _clean_float(row['dyn_tp_mult'], tp_mult, 2)

        unit_v = gk_val if vol_model == "gk" else atr_val
        if unit_v <= 0:
            unit_v = max(c_val * 0.015, 0.01)

        stop_dist = max(d_sl * unit_v, c_val * 0.004, 0.01)
        risk_dollars = initial_capital * risk_frac
        max_pool_dollars = initial_capital * pool_frac
        raw_sh = min(risk_dollars / stop_dist, max_pool_dollars / max(c_val, 0.01))
        sug_shares = max(1, int(round(_clean_float(raw_sh, 1.0))))
        sug_pos_val = _clean_float(sug_shares * c_val, 0.0, 2)
        sug_risk_dlr = _clean_float(sug_shares * stop_dist, 0.0, 2)

        candles.append({"time": ts, "open": o_val, "high": h_val, "low": l_val, "close": c_val})
        volumes.append({
            "time": ts, "value": v_val,
            "color": "rgba(0, 230, 118, 0.35)" if c_val >= o_val else "rgba(255, 23, 68, 0.35)"
        })

        chg_pct = ((c_val - o_val) / max(o_val, 1e-6)) * 100.0
        ema200_v = _clean_float(row['ema200'], c_val, 2)

        bar_details[str(ts)] = {
            "dt": _fmt_dt(dt, tf),
            "o": _clean_float(o_val, 0.0, 2),
            "h": _clean_float(h_val, 0.0, 2),
            "l": _clean_float(l_val, 0.0, 2),
            "c": _clean_float(c_val, 0.0, 2),
            "chg_pct": _clean_float(chg_pct, 0.0, 2),
            "vol": int(v_val),
            "rvol": _clean_float(row['rvol'], 1.0, 2),
            "ema20": _clean_float(row['ema20'], c_val, 2),
            "ema50": _clean_float(row['ema50'], c_val, 2),
            "ema200": ema200_v,
            "vwap": _clean_float(row['vwap'], c_val, 2),
            "rsi": _clean_float(row['rsi'], 50.0, 1),
            "atr": _clean_float(atr_val, 0.0, 2),
            "gk_pct": _clean_float(float(row['gk_vol_pct']) * 100.0, 2.0, 2),
            "score": _clean_float(row['composite_score'], 0.0, 2),
            "dyn_sl_mult": d_sl,
            "dyn_tp_mult": d_tp,
            "unit_vol": _clean_float(unit_v, 0.01, 4),
            "regime": "UNCLEAR / VOLATILE" if bool(row['is_unclear_volatile']) else (
                "BULL TREND" if _clean_float(row['ema20']) >= _clean_float(row['ema50']) else "BEAR TREND"
            ),
            "vs_ema200": "ABOVE EMA200" if c_val >= ema200_v else "BELOW EMA200",
            "sug_shares": sug_shares,
            "sug_pos_val": sug_pos_val,
            "sug_risk": sug_risk_dlr,
            "long_sl": _clean_float(c_val - stop_dist, 0.0, 2),
            "long_tp": _clean_float(c_val + d_tp * unit_v, 0.0, 2),
            "short_sl": _clean_float(c_val + stop_dist, 0.0, 2),
            "short_tp": _clean_float(c_val - d_tp * unit_v, 0.0, 2)
        }

        for col in series_data.keys():
            val = row.get(col, np.nan)
            if not pd.isna(val) and not math.isinf(float(val)):
                series_data[col].append({"time": ts, "value": round(float(val), 4)})

        if bool(row['fvg_bull']) and i >= 2:
            top_v = row['fvg_bull_top']
            bot_v = row['fvg_bull_bot']
            if not pd.isna(top_v) and not pd.isna(bot_v):
                t_end = idx_timestamps[min(i + 8, len(idx_timestamps) - 1)]
                fvg_boxes.append({
                    "t1": idx_timestamps[i - 2], "t2": t_end,
                    "top": _clean_float(top_v), "bot": _clean_float(bot_v), "kind": "BULL_FVG"
                })
        if bool(row['fvg_bear']) and i >= 2:
            top_v = row['fvg_bear_top']
            bot_v = row['fvg_bear_bot']
            if not pd.isna(top_v) and not pd.isna(bot_v):
                t_end = idx_timestamps[min(i + 8, len(idx_timestamps) - 1)]
                fvg_boxes.append({
                    "t1": idx_timestamps[i - 2], "t2": t_end,
                    "top": _clean_float(top_v), "bot": _clean_float(bot_v), "kind": "BEAR_FVG"
                })

        is_unc = bool(row['is_unclear_volatile'])
        if is_unc and not in_unclear:
            in_unclear = True
            unclear_start = ts
        elif not is_unc and in_unclear:
            in_unclear = False
            unclear_spans.append({"t1": unclear_start, "t2": ts})

    if in_unclear and unclear_start is not None:
        unclear_spans.append({"t1": unclear_start, "t2": idx_timestamps[-1]})

    last_row = df_view.iloc[-1]
    last_close = _clean_float(last_row['close'], 100.0, 2)
    last_atr = _clean_float(last_row['atr'], 1.5, 2)
    last_gk_pct = _clean_float(float(last_row['gk_vol_pct']) * 100.0, 2.5, 2)
    last_ema200 = _clean_float(last_row['ema200'], last_close, 2)

    payload = {
        "ticker": ticker.upper(),
        "tf": tf,
        "server_updated_at": datetime.now().strftime("%H:%M:%S"),
        "total_bars": len(candles),
        "first_date": _fmt_dt(df_view.index[0], tf),
        "last_date": _fmt_dt(df_view.index[-1], tf),
        "next_tf": NEXT_TIMEFRAME.get(tf, "1m"),
        "warning": warn_msg,
        "unavailable_indicators": unavailable_list,
        "auto_indicators": STRATEGY_METADATA[strategy]["indicators"],
        "candles": candles,
        "volumes": volumes,
        "bar_details": bar_details,
        "series": series_data,
        "fvg_boxes": fvg_boxes[-60:],
        "unclear_spans": unclear_spans,
        "backtest": bt_selected,
        "all_strategies_summary": all_summary,
        "latest_metrics": {
            "close": last_close,
            "ema200": last_ema200,
            "atr": last_atr,
            "gk_daily_vol_pct": last_gk_pct,
            "composite_score": _clean_float(last_row['composite_score'], 0.0, 2),
            "dyn_sl_mult": sl_mult if use_original else _clean_float(last_row['dyn_sl_mult'], sl_mult, 2),
            "dyn_tp_mult": tp_mult if use_original else _clean_float(last_row['dyn_tp_mult'], tp_mult, 2)
        }
    }
    return JSONResponse(content=_sanitize_for_json(payload))

@app.get("/api/options_greeks")
def api_options_greeks(
    spot: float = Query(100.0),
    strike: float = Query(105.0),
    dte: int = Query(14),
    iv_pct: float = Query(55.0),
    r_pct: float = Query(4.25),
    gk_daily_vol_pct: float = Query(2.5),
    opt_type: str = Query("call")
):
    return black_scholes_analysis(
        S=spot, K=strike, days_to_expiry=dte, iv_pct=iv_pct,
        r_pct=r_pct, gk_daily_vol_pct=gk_daily_vol_pct, opt_type=opt_type
    )

# ==============================================================================
# 6. FRONTEND SINGLE-PAGE APPLICATION
# ==============================================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Quant Lab: Dual-System Re-Entry & Trade Inspector</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {
            --bg: #0b0e14;
            --panel: #131722;
            --border: #2a2e39;
            --text: #d1d4dc;
            --muted: #787b86;
            --accent: #2962ff;
            --green: #00e676;
            --red: #ff1744;
            --gold: #ffd600;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; font-size: 13px; }
        
        #async-loading-banner {
            position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
            background: linear-gradient(90deg, #1e293b, #0f172a, #1e293b);
            border-bottom: 2px solid var(--accent); color: #38bdf8;
            padding: 8px 16px; font-weight: 600; font-size: 12px;
            display: none; align-items: center; justify-content: space-between;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .spinner {
            width: 14px; height: 14px; border: 2px solid rgba(56, 189, 248, 0.3);
            border-top-color: #38bdf8; border-radius: 50%;
            animation: spin 0.8s linear infinite; display: inline-block; margin-right: 8px; vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .sidebar { width: 360px; background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; transition: width 0.2s ease, padding 0.2s ease; flex-shrink: 0; }
        .sidebar.collapsed { width: 0; padding: 0; border-right: none; overflow: hidden; }
        
        details.fold-card { background: #1a1f2e; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }
        details.fold-card > summary { font-weight: 700; font-size: 11.5px; text-transform: uppercase; color: #90a4ae; letter-spacing: 0.5px; cursor: pointer; list-style: none; display: flex; justify-content: space-between; align-items: center; user-select: none; }
        details.fold-card > summary::-webkit-details-marker { display: none; }
        details.fold-card > summary::after { content: "▼"; font-size: 10px; color: var(--muted); transition: transform 0.15s ease; }
        details.fold-card:not([open]) > summary::after { transform: rotate(-90deg); }
        details.fold-card[open] > summary { margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
        
        label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 3px; margin-top: 6px; }
        input[type="text"], input[type="number"], select { width: 100%; background: #0b0e14; border: 1px solid var(--border); color: var(--text); padding: 5px 8px; border-radius: 4px; font-size: 12px; }
        .row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
        .chk-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 4px; }
        .chk-item { display: flex; align-items: center; gap: 6px; font-size: 11px; cursor: pointer; }
        
        button.btn { background: var(--accent); color: #fff; border: none; border-radius: 4px; padding: 6px 10px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 8px; font-size: 12px; }
        button.btn:hover { opacity: 0.9; }
        button.btn-sec { background: #2a2e39; color: var(--text); }
        
        .main { flex: 1; display: flex; flex-direction: column; height: 100vh; overflow-y: auto; }
        .topbar { background: var(--panel); border-bottom: 1px solid var(--border); padding: 6px 12px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; flex-shrink: 0; }
        .left-nav { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .toggle-btn { background: #262b3d; border: 1px solid #363c4e; color: var(--text); padding: 4px 9px; border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: 600; }
        .toggle-btn:hover { background: #323950; }
        .breadcrumbs { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
        .crumb { background: #1e2433; border: 1px solid #363c4e; color: var(--text); padding: 3px 8px; border-radius: 4px; font-size: 11px; display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
        .crumb.active { background: var(--accent); border-color: var(--accent); font-weight: 700; }
        .crumb-close { font-weight: bold; color: #ff8a80; cursor: pointer; padding: 0 3px; border-radius: 3px; font-size: 12px; line-height: 1; }
        .crumb-close:hover { background: rgba(255, 23, 68, 0.35); color: #fff; }
        .crumb-clear { background: rgba(255, 23, 68, 0.15); border: 1px solid rgba(255, 23, 68, 0.4); color: #ff8a80; padding: 3px 7px; border-radius: 4px; font-size: 10.5px; cursor: pointer; }
        
        .tf-selector { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
        .tf-btn { background: #1a1f2e; border: 1px solid var(--border); color: var(--muted); padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 600; }
        .tf-btn.active { background: #00b0ff; color: #000; border-color: #00b0ff; }
        .live-badge { background: rgba(0, 230, 118, 0.14); border: 1px solid rgba(0, 230, 118, 0.45); color: #00e676; padding: 3px 8px; border-radius: 4px; font-size: 10.5px; font-weight: 600; }

        .replay-bar { background: #161b28; border-bottom: 1px solid var(--border); padding: 6px 12px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; flex-shrink: 0; }
        .replay-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .rp-btn { background: #23293a; border: 1px solid #3a4259; color: var(--text); padding: 4px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; }
        .rp-btn:hover { background: #2f374d; }
        .rp-btn.active { background: var(--gold); color: #000; border-color: var(--gold); }
        .rp-buy { background: rgba(0, 230, 118, 0.2); border-color: #00e676; color: #00e676; }
        .rp-buy:hover { background: #00e676; color: #000; }
        .rp-short { background: rgba(255, 23, 68, 0.2); border-color: #ff1744; color: #ff8a80; }
        .rp-short:hover { background: #ff1744; color: #fff; }
        .rp-close { background: rgba(255, 214, 0, 0.18); border-color: #ffd600; color: #ffd600; }
        
        .alert-banner { background: rgba(255, 152, 0, 0.15); border-bottom: 1px solid #ff9800; color: #ffb74d; padding: 5px 14px; font-size: 11.5px; display: none; flex-shrink: 0; }
        
        .chart-wrapper { position: relative; width: 100%; flex: 1; min-height: 420px; background: #0b0e14; border-bottom: 1px solid var(--border); }
        #tv-chart { width: 100%; height: 100%; }
        #overlay-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }
        .chart-legend { position: absolute; top: 10px; left: 14px; z-index: 20; background: rgba(19, 23, 34, 0.88); padding: 6px 10px; border-radius: 4px; border: 1px solid var(--border); pointer-events: none; font-size: 11px; line-height: 1.45; }
        
        .floating-hud { position: absolute; top: 10px; right: 72px; z-index: 25; width: 325px; background: rgba(15, 19, 28, 0.94); border: 1px solid #363c4e; border-radius: 6px; padding: 9px 11px; pointer-events: none; font-size: 11px; line-height: 1.45; box-shadow: 0 6px 18px rgba(0,0,0,0.5); display: none; }
        .hud-header { font-weight: 700; font-size: 11.5px; color: #00b0ff; border-bottom: 1px solid var(--border); padding-bottom: 4px; margin-bottom: 5px; display: flex; justify-content: space-between; }
        .hud-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 10px; }
        .hud-size-box { margin-top: 6px; padding-top: 5px; border-top: 1px dashed #363c4e; color: #ffd600; font-size: 11px; }

        .bottom-panels { padding: 8px 12px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
        .bottom-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 10px; }
        
        table { width: 100%; border-collapse: collapse; font-size: 11px; }
        th, td { padding: 5px 7px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
        th { color: var(--muted); font-weight: 600; background: #161b26; position: sticky; top: 0; z-index: 2; }
        tr.highlight-row { background: rgba(41, 98, 255, 0.22); font-weight: 600; }
        tr.selected-trade-row { background: rgba(0, 176, 255, 0.28) !important; outline: 1px solid #00b0ff; }
        tr:hover { background: rgba(255, 255, 255, 0.05); }
        
        .pos { color: var(--green); }
        .neg { color: var(--red); }
        .pill { padding: 2px 5px; border-radius: 3px; font-size: 10px; font-weight: 700; }
        .pill-cheap { background: rgba(0,230,118,0.18); color: #00e676; }
        .pill-exp { background: rgba(255,23,68,0.18); color: #ff1744; }
        .pill-fair { background: rgba(255,214,0,0.15); color: #ffd600; }
        
        .strategy-guide-box { font-size: 11px; line-height: 1.45; background: #121722; border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; margin-top: 6px; color: #cbd5e1; }
    </style>
</head>
<body>

<div id="async-loading-banner">
    <div><span class="spinner"></span><span id="async-loading-text">Loading chart & recalculating indicators asynchronously...</span></div>
    <div id="async-elapsed-timer" style="font-family: monospace; color: #38bdf8;">0.0s</div>
</div>

<div class="sidebar" id="sidebar">
    <details class="fold-card" open>
        <summary>1. Asset, Side, Engine Mode & Click Action</summary>
        <div class="row-2">
            <div>
                <label>Ticker (e.g. SNDK, NVDA, ASTS)</label>
                <input type="text" id="ticker" value="SNDK">
            </div>
            <div>
                <label>Trade Direction</label>
                <select id="trade_side" onchange="asyncActionWrapper(loadChartData)">
                    <option value="both" selected>Long & Short (Both)</option>
                    <option value="long">Long Only</option>
                    <option value="short">Short Only</option>
                </select>
            </div>
        </div>
        <label style="color:#00e5ff; font-weight:600;">System SL/TP Engine Mode</label>
        <select id="engine_mode" onchange="asyncActionWrapper(loadChartData)">
            <option value="dynamic" selected>New Dynamic System (Adaptive Volatility + Continuous Trailing Stops)</option>
            <option value="original">Original Code Mode (Fixed Multipliers & Standard Rules)</option>
        </select>
        <label style="color:#ffd600; font-weight:600;">EMA 200 Trend Filter (Applies to All Strategies)</label>
        <select id="ema200_filter" onchange="asyncActionWrapper(loadChartData)">
            <option value="none" selected>No EMA 200 Filter (Allow All Signals)</option>
            <option value="trend_follow">Trend-Following: Ignore Long &lt; EMA200 &amp; Ignore Short &gt; EMA200</option>
            <option value="counter_trend">Counter-Trend: Only Short &gt; EMA200 (Uptrend) &amp; Long &lt; EMA200 (Downtrend)</option>
        </select>
        <label>Candle Click Action</label>
        <select id="clickMode">
            <option value="drill">Drill-Down Timeframe (1Y->1M->1W->1D->1H->15m->5m->1m)</option>
            <option value="replay_here">Start Bar-by-Bar Replay From Clicked Candle</option>
            <option value="manual_long">Single-Click Manual BUY (Long TP/SL)</option>
            <option value="manual_short">Single-Click Manual SHORT (Short TP/SL)</option>
            <option value="cutoff">Set Point-in-Time Cutoff Date</option>
        </select>
        <div class="row-2" style="margin-top:4px;">
            <button class="btn" onclick="asyncActionWrapper(resetToRoot)">Load Max History</button>
            <button class="btn btn-sec" onclick="asyncActionWrapper(clearCutoffAndManual)">Reset Zoom/Replay</button>
        </div>
    </details>

    <details class="fold-card" open>
        <summary>2. Strategy & Continuous Re-Entry Formula</summary>
        <label>Active Strategy / Rule</label>
        <select id="strategy" onchange="onStrategyChange()">
            <option value="trend">1. Core Trend (EMA 20/50/200 + Volume L/S)</option>
            <option value="watch_dip">2. Watch -> Pullback/Rally -> Buy/Short</option>
            <option value="smc_fvg">3. SMC & Fair Value Gaps (Bull/Bear FVG)</option>
            <option value="fib_golden">4. Fibonacci Golden Pocket (0.618 / 0.382)</option>
            <option value="orb">5. Opening Range Breakout/Breakdown (ORB)</option>
            <option value="do_t">6. Intraday 'Do T' (VWAP +-2 Sigma L/S)</option>
            <option value="composite">7. Composite Score (a*A + b*B + c*C)</option>
            <option value="scale_ladder">8. Momentum Scale-Out Ladder (Sell xx% @ +xx%)</option>
            <option value="plain_tpsl">9. Plain TP/SL & Manual Replay Sandbox</option>
            <option value="no_tp_trend" selected>10. Dynamic Trailing SL / No Static TP (Trend Hold)</option>
        </select>
        <div class="strategy-guide-box" id="strategy-guide-text">Loading strategy guide...</div>
        
        <div class="row-2" style="margin-top:6px;">
            <div>
                <label>SL/TP Mode</label>
                <select id="dynamic_tpsl" onchange="asyncActionWrapper(loadChartData)">
                    <option value="true" selected>Dynamic Formula (Regime+Vol)</option>
                    <option value="false">Fixed Multipliers</option>
                </select>
            </div>
            <div>
                <label>Band Volatility Model</label>
                <select id="vol_model" onchange="asyncActionWrapper(loadChartData)">
                    <option value="atr">ATR (14)</option>
                    <option value="gk">Garman-Klass</option>
                </select>
            </div>
        </div>
        <div class="row-2" style="margin-top:4px;">
            <div>
                <label>Base SL Mult (Anchor)</label>
                <input type="number" id="sl_mult" value="1.5" step="0.25">
            </div>
            <div>
                <label>Base TP Mult (Anchor)</label>
                <input type="number" id="tp_mult" value="2.8" step="0.25">
            </div>
        </div>
        <label style="margin-top:8px;">Composite Weights (Alpha / Beta / Gamma)</label>
        <div class="row-3">
            <div><label>a (Trend)</label><input type="number" id="w_trend" value="0.4" step="0.1"></div>
            <div><label>b (Liquid)</label><input type="number" id="w_liq" value="0.4" step="0.1"></div>
            <div><label>g (VWAP)</label><input type="number" id="w_rev" value="0.2" step="0.1"></div>
        </div>
        <button class="btn" onclick="asyncActionWrapper(loadChartData)">Apply Strategy & Formula</button>
    </details>

    <details class="fold-card" open>
        <summary>3. Scale-Out Profit Ladder (All Strategies)</summary>
        <label class="chk-item" style="margin-bottom:6px; color:#00e676; font-weight:600;">
            <input type="checkbox" id="enable_scale_out" checked onchange="asyncActionWrapper(loadChartData)"> Apply Multi-Tranche Scale-Out to Every Strategy & Replay
        </label>
        <div class="row-2">
            <div>
                <label>Tier 1 Trigger (+Gain %)</label>
                <input type="number" id="t1_gain_pct" value="4.0" step="0.5" onchange="asyncActionWrapper(loadChartData)">
            </div>
            <div>
                <label>Tier 1 Sell (% Shares)</label>
                <input type="number" id="t1_sell_pct" value="35" step="5" onchange="asyncActionWrapper(loadChartData)">
            </div>
        </div>
        <div class="row-2">
            <div>
                <label>Tier 2 Trigger (+Gain %)</label>
                <input type="number" id="t2_gain_pct" value="8.0" step="0.5" onchange="asyncActionWrapper(loadChartData)">
            </div>
            <div>
                <label>Tier 2 Sell (% Shares)</label>
                <input type="number" id="t2_sell_pct" value="35" step="5" onchange="asyncActionWrapper(loadChartData)">
            </div>
        </div>
        <div style="font-size:10.5px; color:var(--muted); margin-top:5px;">
            * Remaining shares ride to Dynamic TP with Stop-Loss ratcheted to Breakeven (at T1) and T1 Price (at T2).
        </div>
    </details>

    <details class="fold-card" open>
        <summary>4. Dual-Pool Portfolio & Replay Allocation</summary>
        <div class="row-2">
            <div>
                <label>Starting Capital ($)</label>
                <input type="number" id="initial_capital" value="100000" step="10000" onchange="asyncActionWrapper(loadChartData)">
            </div>
            <div>
                <label>Auto Risk / Trade (%)</label>
                <input type="number" id="risk_pct" value="1.25" step="0.25" onchange="asyncActionWrapper(loadChartData)">
            </div>
        </div>
        <div class="row-2">
            <div>
                <label>Core Trend Pool (%)</label>
                <input type="number" id="core_pct" value="70" step="5" onchange="asyncActionWrapper(loadChartData)">
            </div>
            <div>
                <label>Intraday / T Pool (%)</label>
                <input type="number" id="intraday_pct" value="30" step="5" onchange="asyncActionWrapper(loadChartData)">
            </div>
        </div>
        <div style="margin-top:6px; padding-top:6px; border-top:1px dashed var(--border);">
            <label style="color:#00e5ff; font-weight:600;">Manual Bar-by-Bar Replay Trade Allocation (% of Total Portfolio)</label>
            <input type="number" id="manual_alloc_pct" value="25" min="1" max="100" step="5" onchange="syncAllocInput(this.value)">
        </div>
    </details>

    <details class="fold-card">
        <summary>5. Selectable Overlays & Zones</summary>
        <div class="chk-grid">
            <label class="chk-item"><input type="checkbox" id="chk_ema20" checked onchange="renderOverlays()"> EMA 20 & 50</label>
            <label class="chk-item"><input type="checkbox" id="chk_ema200" checked onchange="renderOverlays()"> EMA 200 (Gold)</label>
            <label class="chk-item"><input type="checkbox" id="chk_vwap" onchange="renderOverlays()"> Session VWAP</label>
            <label class="chk-item"><input type="checkbox" id="chk_vwap_bands" onchange="renderOverlays()"> VWAP +-2s Bands</label>
            <label class="chk-item"><input type="checkbox" id="chk_atr_bands" checked onchange="renderOverlays()"> Dyn SL ATR Band</label>
            <label class="chk-item"><input type="checkbox" id="chk_gk_bands" checked onchange="renderOverlays()"> Dyn TP GK Cone</label>
            <label class="chk-item"><input type="checkbox" id="chk_fib_levels" onchange="renderOverlays()"> Fibonacci Levels</label>
            <label class="chk-item"><input type="checkbox" id="chk_orb_levels" onchange="renderOverlays()"> ORB High/Low</label>
            <label class="chk-item"><input type="checkbox" id="chk_fvg_zones" checked onchange="renderOverlays()"> SMC FVG Boxes</label>
            <label class="chk-item"><input type="checkbox" id="chk_tpsl_boxes" checked onchange="renderOverlays()"> Shaded TP/SL Boxes</label>
            <label class="chk-item"><input type="checkbox" id="chk_unclear" checked onchange="renderOverlays()"> Gray Volatile Area</label>
            <label class="chk-item"><input type="checkbox" id="chk_hover_hud" checked> Floating Hover HUD</label>
        </div>
    </details>
</div>

<div class="main">
    <div class="topbar">
        <div class="left-nav">
            <button class="toggle-btn" id="btn-sidebar" onclick="toggleSidebar()">◀ Hide Controls</button>
            <button class="toggle-btn" id="btn-panels" onclick="toggleAllPanels()">▼ Fold All Panels</button>
            <button class="toggle-btn" onclick="fitFullChart()" title="Reset zoom to show all bars">🔍 Fit Full Chart</button>
            <div class="breadcrumbs" id="breadcrumb-bar"></div>
        </div>
        <div class="tf-selector" id="tf-buttons">
            <span class="live-badge" id="live-refresh-badge" title="1m chart data & 60-day Option Chain auto-refresh every 60s">⟳ 1m & Options Sync: 60s</span>
            <button class="tf-btn active" onclick="asyncActionWrapper(() => switchTF('1Y'))">1Y</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('1M'))">1M</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('1W'))">1W</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('1d'))">1D</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('1h'))">1H</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('15m'))">15m</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('5m'))">5m</button>
            <button class="tf-btn" onclick="asyncActionWrapper(() => switchTF('1m'))">1m</button>
        </div>
    </div>

    <div class="replay-bar">
        <div class="replay-controls">
            <span style="font-weight:700; color:#00b0ff; font-size:11.5px;">🎬 BAR-BY-BAR REPLAY:</span>
            <button class="rp-btn" id="btn-replay-start" onclick="startBarReplay()">⏮ Start Replay</button>
            <button class="rp-btn" onclick="stepReplay(-1)" title="Step Back 1 Stick (Backspace)">◀ -1 Bar</button>
            <button class="rp-btn" onclick="stepReplay(1)" title="Advance Forward 1 Stick (Enter)">▶ +1 Bar (Next Stick)</button>
            <button class="rp-btn" id="btn-replay-play" onclick="toggleReplayAutoPlay()">⏯ Auto-Play</button>
            <select id="replay_speed" style="width:78px; padding:3px 5px; font-size:11px;" onchange="if(replayTimer){toggleReplayAutoPlay(); toggleReplayAutoPlay();}">
                <option value="800">0.8s / bar</option>
                <option value="400" selected>0.4s / bar</option>
                <option value="180">0.18s / bar</option>
            </select>
            <button class="rp-btn" onclick="stopBarReplay()">⏹ Exit Replay</button>
            <span id="replay-status-text" style="color:var(--gold); font-weight:600; font-size:11px; margin-left:4px;">[Full History Mode]</span>
        </div>
        <div class="replay-controls">
            <span style="font-size:11px; color:var(--muted);">Order Size (% Total Equity):</span>
            <input type="number" id="top_alloc_pct" value="25" min="1" max="100" step="5" style="width:62px; padding:3px 6px;" onchange="syncAllocInput(this.value)">
            <span style="font-size:11px; color:var(--muted);">%</span>
            <button class="rp-btn rp-buy" onclick="executeManualReplayOrder('LONG')">▲ BUY LONG (@ % Equity)</button>
            <button class="rp-btn rp-short" onclick="executeManualReplayOrder('SHORT')">▼ SELL SHORT (@ % Equity)</button>
            <button class="rp-btn rp-close" id="btn-close-manual" onclick="closeManualReplayPosition()" style="display:none;">✖ CLOSE OPEN POS</button>
            <button class="rp-btn" onclick="clearManualReplayHistory()" title="Clear manual replay trades">🗑 Reset Manual</button>
        </div>
    </div>

    <div class="alert-banner" id="alert-banner"></div>

    <div class="chart-wrapper" id="chart-wrap">
        <div class="chart-legend" id="chart-legend">Loading...</div>
        <div class="floating-hud" id="floating-hud"></div>
        <div id="tv-chart"></div>
        <canvas id="overlay-canvas"></canvas>
    </div>

    <div class="bottom-panels">
        <!-- LIVE TRADE INSPECTOR DRAWER -->
        <details class="fold-card bottom-fold" id="inspector-drawer" style="display:none; border-color:#00b0ff;">
            <summary style="color:#00b0ff;">
                <span>🔍 Live Trade Inspector (Focused Intra-Trade Zoom View)</span>
                <span id="inspector-status" style="font-size:11px; color:#ffd600; margin-right:10px;"></span>
            </summary>
            <div style="width:100%; height:180px;" id="inspector-chart-container"></div>
        </details>

        <details class="fold-card bottom-fold" open ontoggle="resizeChart()">
            <summary>
                <span>Active Strategy & Bar-by-Bar Replay Trade Execution Log (Click Any Row to Smart-Zoom Trade on Chart)</span>
                <span id="trade-count-badge" style="color:#00e676; margin-right:10px;"></span>
            </summary>
            <div style="overflow-x:auto; max-height:195px;">
                <table id="trade-log-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Side</th>
                            <th>Signal (Dyn SL/TP)</th>
                            <th>Bucket Pool</th>
                            <th>Entry Time</th>
                            <th>Entry $</th>
                            <th>Initial Shares</th>
                            <th>Position Value (% Eq)</th>
                            <th>Dollar Risk</th>
                            <th>SL / TP ($)</th>
                            <th>Scale-Out Ladder</th>
                            <th>Exit Time</th>
                            <th>Exit $</th>
                            <th>Result</th>
                            <th>Net PnL ($)</th>
                            <th>Return %</th>
                            <th>Total Equity</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </details>

        <div class="bottom-grid">
            <details class="fold-card bottom-fold" open ontoggle="resizeChart()">
                <summary>
                    <span>All Strategies Comparison ($100k Dual-Pool Baseline)</span>
                    <span id="bh-badge" style="color:var(--gold); margin-right:10px;"></span>
                </summary>
                <div style="overflow-x:auto; max-height:190px;">
                    <table id="summary-table">
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Pool</th>
                                <th>Ending Equity</th>
                                <th>Net Return</th>
                                <th>Trades (L / S)</th>
                                <th>Win Rate</th>
                                <th>Profit Factor</th>
                                <th>Max DD</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </details>

            <details class="fold-card bottom-fold" open ontoggle="resizeChart()">
                <summary>Options Greeks & Implied Expectancy Calculator</summary>
                <div class="row-3">
                    <div><label>Spot ($)</label><input type="number" id="opt_spot" value="100" step="1"></div>
                    <div><label>Strike ($)</label><input type="number" id="opt_strike" value="105" step="1"></div>
                    <div><label>Type</label>
                        <select id="opt_type"><option value="call">CALL</option><option value="put">PUT</option></select>
                    </div>
                </div>
                <div class="row-3">
                    <div><label>DTE (Days)</label><input type="number" id="opt_dte" value="14"></div>
                    <div><label>Implied Vol (%)</label><input type="number" id="opt_iv" value="55"></div>
                    <div><label>GK Daily Vol (%)</label><input type="number" id="opt_gk" value="2.5" readonly></div>
                </div>
                <button class="btn" onclick="calcOptions()">Evaluate Option & Stock Expectancy</button>
                <div id="opt-results" style="margin-top:6px; font-size:11px; line-height:1.5;"></div>
            </details>
        </div>

        <details class="fold-card bottom-fold" open ontoggle="resizeChart()">
            <summary>
                <span>Live Option Chain Browser (Up to 60 Days DTE — Auto-Updates Every 1m) & 60-Day Contract History</span>
                <span id="chain-source" style="color:var(--muted); font-weight:normal; font-size:10.5px; margin-right:10px;"></span>
            </summary>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:11px; color:#80d8ff;">Showing all Option Expirations up to 60 Days Out (Click any strike to inspect 60-day history & Greeks)</span>
                <div style="display:flex; gap:6px; align-items:center;">
                    <span style="font-size:11px; color:var(--muted);">Expiration (<= 60d):</span>
                    <select id="chain_expiry" style="width:160px;" onchange="asyncActionWrapper(() => loadOptionChain(this.value))"></select>
                </div>
            </div>
            <div class="row-2" style="gap:14px;">
                <div style="max-height:190px; overflow-y:auto;">
                    <div style="font-weight:700; color:var(--green); margin-bottom:4px;">CALLS (Click row for 60d contract history)</div>
                    <table id="calls-table">
                        <thead>
                            <tr><th>Strike</th><th>DTE</th><th>Last</th><th>Bid/Ask</th><th>IV%</th><th>Delta</th><th>Vol/OI</th><th>Valuation</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div style="max-height:190px; overflow-y:auto;">
                    <div style="font-weight:700; color:var(--red); margin-bottom:4px;">PUTS (Click row for 60d contract history)</div>
                    <table id="puts-table">
                        <thead>
                            <tr><th>Strike</th><th>DTE</th><th>Last</th><th>Bid/Ask</th><th>IV%</th><th>Delta</th><th>Vol/OI</th><th>Valuation</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            <div id="contract-history-box" style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border); display:none;">
                <div style="font-weight:700; margin-bottom:6px; color:#00b0ff;" id="contract-hist-title"></div>
                <div id="contract-hist-strip" style="display:flex; gap:6px; overflow-x:auto; padding-bottom:4px;"></div>
            </div>
        </details>
    </div>
</div>

<script>
let chart, candleSeries, volumeSeries;
let lineSeriesMap = {};
let inspectorChart = null;
let inspectorCandleSeries = null;

let currentData = null;
let currentTF = "1Y";
let currentStartTs = null;
let currentCutoffTs = null;
let manualEntryTs = null;
let manualSide = "long";
let focusedTradeNo = null;
let breadcrumbs = [{ label: "Max (1Y)", tf: "1Y", startTs: null }];
let activeCrumbIdx = 0;
let allPanelsFolded = false;

let asyncTimerInterval = null;
let asyncStartTime = 0;

async function asyncActionWrapper(asyncFn) {
    const banner = document.getElementById('async-loading-banner');
    const timerEl = document.getElementById('async-elapsed-timer');
    banner.style.display = 'flex';
    asyncStartTime = performance.now();
    timerEl.textContent = '0.0s';
    
    if (asyncTimerInterval) clearInterval(asyncTimerInterval);
    asyncTimerInterval = setInterval(() => {
        const elapsed = ((performance.now() - asyncStartTime) / 1000).toFixed(1);
        timerEl.textContent = elapsed + 's';
    }, 100);

    try {
        await asyncFn();
    } catch (err) {
        console.error(err);
    } finally {
        if (asyncTimerInterval) clearInterval(asyncTimerInterval);
        banner.style.display = 'none';
    }
}

const STRATEGY_GUIDES = {
    "trend": "1. Core Trend: Rides established macro trends by entering on pullbacks toward EMA 20 in the direction of the EMA 50/200 with supportive volume.",
    "watch_dip": "2. Watch -> Pullback/Rally: Monitors trend momentum, waits for a measured pullback/rally (>= 0.8x volatility), and executes on reversal candle confirmation.",
    "smc_fvg": "3. SMC & Fair Value Gaps: Institutional Smart Money Concepts strategy detecting liquidity imbalances and trading the subsequent retest of Bull/Bear FVGs.",
    "fib_golden": "4. Fibonacci Golden Pocket: Automatically traces market swings, buying the 0.618 golden pocket in bull regimes or shorting the 0.382 zone in bear regimes.",
    "orb": "5. Opening Range Breakout: Intraday momentum strategy capturing strong breakouts beyond the first 30-minute high/low range with high relative volume.",
    "do_t": "6. Intraday 'Do T': Mean-reversion strategy fading extreme intraday extensions beyond VWAP +-2 standard deviation bands, targeting reversion to VWAP.",
    "composite": "7. Composite Score: Multi-factor quantitative alpha score combining trend slope, price pressure liquidity, and VWAP z-scores.",
    "scale_ladder": "8. Momentum Scale-Out Ladder: Trend-following with multi-tier partial profit-taking (+4%, +8%) and stop-loss ratcheting.",
    "plain_tpsl": "9. Plain TP/SL Sandbox: Manual sandbox mode for testing custom entries via chart clicks or fixed risk/reward parameters.",
    "no_tp_trend": "10. Dynamic Trailing SL / No Static TP: Rule/Strategy that holds positions indefinitely without setting a static Take-Profit. Ratchets the trailing stop-loss continuously behind price action until trend reversal."
};

function onStrategyChange() {
    const stratKey = document.getElementById('strategy').value;
    document.getElementById('strategy-guide-text').textContent = STRATEGY_GUIDES[stratKey] || '';
    asyncActionWrapper(loadChartData);
}

let replayActive = false;
let replayIndex = 0;
let replayTimer = null;
let manualReplayOrders = [];
let manualSimResult = null;

let liveRefreshCountdown = 60;
let lastSelectedOptionContract = null;

function syncAllocInput(val) {
    const clean = Math.max(1, Math.min(100, parseFloat(val) || 25));
    document.getElementById('manual_alloc_pct').value = clean;
    document.getElementById('top_alloc_pct').value = clean;
}

function toggleSidebar() {
    const sb = document.getElementById('sidebar');
    const btn = document.getElementById('btn-sidebar');
    sb.classList.toggle('collapsed');
    btn.textContent = sb.classList.contains('collapsed') ? '▶ Show Controls' : '◀ Hide Controls';
    setTimeout(resizeChart, 220);
}

function toggleAllPanels() {
    allPanelsFolded = !allPanelsFolded;
    document.querySelectorAll('details.bottom-fold').forEach(d => {
        if (allPanelsFolded) d.removeAttribute('open');
        else d.setAttribute('open', '');
    });
    document.getElementById('btn-panels').textContent = allPanelsFolded ? '▲ Expand Panels' : '▼ Fold All Panels';
    setTimeout(resizeChart, 60);
}

function fitFullChart() {
    focusedTradeNo = null;
    document.querySelectorAll('#trade-log-table tbody tr').forEach(r => r.classList.remove('selected-trade-row'));
    if (chart) {
        chart.priceScale('right').applyOptions({ autoScale: true });
        chart.timeScale().fitContent();
        requestAnimationFrame(drawCanvasZones);
    }
}

function resizeChart() {
    if (!chart) return;
    const wrap = document.getElementById('chart-wrap');
    chart.applyOptions({ width: wrap.clientWidth, height: wrap.clientHeight });
    requestAnimationFrame(drawCanvasZones);
    if (inspectorChart) {
        const insWrap = document.getElementById('inspector-chart-container');
        if (insWrap) inspectorChart.applyOptions({ width: insWrap.clientWidth, height: insWrap.clientHeight });
    }
}

function initChart() {
    const container = document.getElementById('tv-chart');
    chart = LightweightCharts.createChart(container, {
        layout: { background: { type: 'solid', color: '#0b0e14' }, textColor: '#d1d4dc' },
        grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#2a2e39', autoScale: true },
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false }
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: '#00e676', downColor: '#ff1744',
        borderUpColor: '#00e676', borderDownColor: '#ff1744',
        wickUpColor: '#00e676', wickDownColor: '#ff1744'
    });

    volumeSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    const lineConfigs = {
        ema20:      { color: '#2962ff', lineWidth: 2, title: 'EMA 20' },
        ema50:      { color: '#ff9800', lineWidth: 2, title: 'EMA 50' },
        ema200:     { color: '#ffd600', lineWidth: 2, title: 'EMA 200' },
        vwap:       { color: '#e040fb', lineWidth: 2, title: 'VWAP' },
        vwap_u2:    { color: 'rgba(224,64,251,0.5)', lineWidth: 1, lineStyle: 2, title: 'VWAP +2s' },
        vwap_l2:    { color: 'rgba(224,64,251,0.5)', lineWidth: 1, lineStyle: 2, title: 'VWAP -2s' },
        atr_upper:  { color: 'rgba(0,229,255,0.6)', lineWidth: 1, lineStyle: 2, title: 'Dyn SL Upper' },
        atr_lower:  { color: 'rgba(0,229,255,0.6)', lineWidth: 1, lineStyle: 2, title: 'Dyn SL Lower' },
        gk_upper:   { color: 'rgba(255,214,0,0.7)', lineWidth: 1, lineStyle: 1, title: 'Dyn TP Upper' },
        gk_lower:   { color: 'rgba(255,214,0,0.7)', lineWidth: 1, lineStyle: 1, title: 'Dyn TP Lower' },
        fib_382:    { color: 'rgba(144,164,174,0.5)', lineWidth: 1, title: 'Fib 0.382' },
        fib_500:    { color: 'rgba(144,164,174,0.5)', lineWidth: 1, title: 'Fib 0.500' },
        fib_618:    { color: '#ffd600', lineWidth: 2, title: 'Fib 0.618 Golden' },
        fib_786:    { color: 'rgba(255,23,68,0.5)', lineWidth: 1, title: 'Fib 0.786' },
        orb_high:   { color: '#00e676', lineWidth: 1, lineStyle: 2, title: 'ORB High' },
        orb_low:    { color: '#ff1744', lineWidth: 1, lineStyle: 2, title: 'ORB Low' }
    };

    for (const [k, cfg] of Object.entries(lineConfigs)) {
        lineSeriesMap[k] = chart.addLineSeries({ ...cfg, lastValueVisible: false, priceLineVisible: false });
    }

    // Init Trade Inspector Sub-Chart
    const insContainer = document.getElementById('inspector-chart-container');
    inspectorChart = LightweightCharts.createChart(insContainer, {
        layout: { background: { type: 'solid', color: '#0b0e14' }, textColor: '#d1d4dc' },
        grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
        rightPriceScale: { borderColor: '#2a2e39', autoScale: true },
        timeScale: { borderColor: '#2a2e39', timeVisible: true }
    });
    inspectorCandleSeries = inspectorChart.addCandlestickSeries({
        upColor: '#00e676', downColor: '#ff1744',
        borderUpColor: '#00e676', borderDownColor: '#ff1744',
        wickUpColor: '#00e676', wickDownColor: '#ff1744'
    });

    chart.timeScale().subscribeVisibleTimeRangeChange(() => requestAnimationFrame(drawCanvasZones));
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(drawCanvasZones));
    window.addEventListener('resize', resizeChart);

    window.addEventListener('keydown', (e) => {
        if (!replayActive) return;
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (activeTag === 'input' || activeTag === 'select' || activeTag === 'textarea') return;

        if (e.key === 'Backspace') {
            e.preventDefault();
            stepReplay(-1);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            stepReplay(1);
        }
    });

    chart.subscribeCrosshairMove((param) => {
        const hud = document.getElementById('floating-hud');
        const showHud = document.getElementById('chk_hover_hud').checked;
        if (!showHud || !param || !param.time || !currentData || !currentData.bar_details) {
            hud.style.display = 'none';
            return;
        }
        const b = currentData.bar_details[String(param.time)];
        if (!b) {
            hud.style.display = 'none';
            return;
        }
        hud.style.display = 'block';
        const chgColor = b.chg_pct >= 0 ? '#00e676' : '#ff1744';
        const regColor = b.regime.includes('UNCLEAR') ? '#ffb74d' : (b.regime.includes('BULL') ? '#00e676' : '#ff1744');
        const ema200Color = b.vs_ema200.includes('ABOVE') ? '#00e676' : '#ff1744';
        hud.innerHTML = `
            <div class="hud-header">
                <span>${currentData.ticker} [${currentData.tf}] — ${b.dt}</span>
                <span style="color:${chgColor}">${b.chg_pct >= 0 ? '+' : ''}${b.chg_pct}%</span>
            </div>
            <div class="hud-grid">
                <div><b>Open:</b> $${b.o}</div>
                <div><b>High:</b> $${b.h}</div>
                <div><b>Low:</b> $${b.l}</div>
                <div><b>Close:</b> $${b.c}</div>
                <div><b>Volume:</b> ${b.vol.toLocaleString()}</div>
                <div><b>RVOL:</b> ${b.rvol}x</div>
                <div><b>EMA 20/50:</b> $${b.ema20} / $${b.ema50}</div>
                <div><b>EMA 200:</b> <span style="color:#ffd600;">$${b.ema200}</span></div>
                <div><b>VWAP:</b> $${b.vwap}</div>
                <div><b>ATR / GK:</b> $${b.atr} / ${b.gk_pct}%</div>
                <div><b>RSI (14):</b> ${b.rsi}</div>
                <div><b>Score:</b> ${b.score}</div>
            </div>
            <div style="margin-top:4px;">
                <b>Regime:</b> <span style="color:${regColor}; font-weight:700;">${b.regime}</span> | <span style="color:${ema200Color}; font-weight:700;">${b.vs_ema200}</span>
            </div>
            <div class="hud-size-box">
                <div><b>Auto Size @ Bar:</b> <b>${b.sug_shares.toLocaleString()} sh</b> ($${b.sug_pos_val.toLocaleString()}) | <b>Dyn:</b> ${b.dyn_sl_mult}x</div>
                <div><b>Long SL:</b> $${b.long_sl} | <b>Short SL:</b> $${b.short_sl}</div>
            </div>
        `;
    });

    chart.subscribeClick((param) => {
        if (!param || !param.time || !currentData) return;
        const clickedTs = param.time;
        const mode = document.getElementById('clickMode').value;
        const dtObj = new Date(clickedTs * 1000);
        const dtStr = dtObj.toISOString().slice(0, 10);

        if (mode === 'drill') {
            const nextTf = currentData.next_tf;
            if (nextTf === currentTF && currentTF === '1m') return;
            if (currentTF === '1Y') {
                const yr = dtObj.getUTCFullYear();
                currentStartTs = Math.floor(Date.UTC(yr, 0, 1) / 1000);
            } else if (currentTF === '1M') {
                const yr = dtObj.getUTCFullYear();
                const mo = dtObj.getUTCMonth();
                currentStartTs = Math.floor(Date.UTC(yr, mo, 1) / 1000);
            } else {
                currentStartTs = clickedTs;
            }
            currentTF = nextTf;
            breadcrumbs = breadcrumbs.slice(0, activeCrumbIdx + 1);
            const existingIdx = breadcrumbs.findIndex(b => b.tf === currentTF);
            const newTab = { label: `${dtStr} (${currentTF})`, tf: currentTF, startTs: currentStartTs };
            if (existingIdx !== -1) {
                breadcrumbs[existingIdx] = newTab;
                breadcrumbs = breadcrumbs.slice(0, existingIdx + 1);
            } else {
                breadcrumbs.push(newTab);
            }
            activeCrumbIdx = breadcrumbs.length - 1;
            asyncActionWrapper(loadChartData);
        } else if (mode === 'replay_here') {
            const idx = currentData.candles.findIndex(c => c.time === clickedTs);
            if (idx !== -1) {
                startBarReplay(idx);
            }
        } else if (mode === 'manual_long') {
            document.getElementById('strategy').value = 'plain_tpsl';
            manualEntryTs = clickedTs;
            manualSide = 'long';
            asyncActionWrapper(loadChartData);
        } else if (mode === 'manual_short') {
            document.getElementById('strategy').value = 'plain_tpsl';
            manualEntryTs = clickedTs;
            manualSide = 'short';
            asyncActionWrapper(loadChartData);
        } else if (mode === 'cutoff') {
            currentCutoffTs = clickedTs;
            asyncActionWrapper(loadChartData);
        }
    });
}

function startBarReplay(startIdx = null) {
    if (!currentData || !currentData.candles || currentData.candles.length < 1) return;
    replayActive = true;
    if (startIdx !== null && startIdx >= 0 && startIdx < currentData.candles.length) {
        replayIndex = startIdx;
    } else {
        replayIndex = 0;
    }
    document.getElementById('btn-replay-start').classList.add('active');
    renderReplayFrame(true);
}

function stepReplay(delta) {
    if (!currentData || !currentData.candles || currentData.candles.length === 0) return;
    if (!replayActive) {
        startBarReplay();
        return;
    }
    const nextIdx = Math.max(0, Math.min(currentData.candles.length - 1, replayIndex + delta));
    replayIndex = nextIdx;
    renderReplayFrame(false);
}

function toggleReplayAutoPlay() {
    const btn = document.getElementById('btn-replay-play');
    if (replayTimer) {
        clearInterval(replayTimer);
        replayTimer = null;
        btn.classList.remove('active');
        btn.textContent = '⏯ Auto-Play';
        return;
    }
    if (!replayActive) startBarReplay();
    const speed = parseInt(document.getElementById('replay_speed').value || '400', 10);
    btn.classList.add('active');
    btn.textContent = '⏸ Pause';
    replayTimer = setInterval(() => {
        if (!currentData || replayIndex >= currentData.candles.length - 1) {
            toggleReplayAutoPlay();
            return;
        }
        stepReplay(1);
    }, speed);
}

function stopBarReplay() {
    if (replayTimer) toggleReplayAutoPlay();
    replayActive = false;
    document.getElementById('btn-replay-start').classList.remove('active');
    document.getElementById('replay-status-text').textContent = '[Full History Mode]';
    document.getElementById('inspector-drawer').style.display = 'none';
    if (!currentData) return;
    replayIndex = currentData.candles.length - 1;
    renderReplayFrame(true);
}

function clearManualReplayHistory() {
    manualReplayOrders = [];
    manualSimResult = null;
    document.getElementById('btn-close-manual').style.display = 'none';
    if (replayActive) {
        renderReplayFrame(false);
    } else if (currentData) {
        candleSeries.setMarkers(currentData.backtest.markers || []);
        renderOverlays();
        updateTradeLogTable(currentData);
    }
}

function executeManualReplayOrder(side) {
    if (!currentData || !currentData.candles || currentData.candles.length === 0) return;
    if (!replayActive) {
        startBarReplay(0);
    }
    recomputeManualReplaySim();
    if (manualSimResult && manualSimResult.hasOpenPosition) {
        const lastOrd = manualReplayOrders[manualReplayOrders.length - 1];
        if (lastOrd && lastOrd.forceCloseIdx === null) {
            lastOrd.forceCloseIdx = replayIndex;
        }
    }
    const allocPct = Math.max(1, Math.min(100, parseFloat(document.getElementById('top_alloc_pct').value) || 25));
    manualReplayOrders.push({ entryIdx: replayIndex, side: side, allocPct: allocPct, forceCloseIdx: null });
    renderReplayFrame(false);
}

function closeManualReplayPosition() {
    if (!replayActive || manualReplayOrders.length === 0) return;
    const lastOrd = manualReplayOrders[manualReplayOrders.length - 1];
    if (lastOrd && lastOrd.forceCloseIdx === null && replayIndex > lastOrd.entryIdx) {
        lastOrd.forceCloseIdx = replayIndex;
        renderReplayFrame(false);
    }
}

function recomputeManualReplaySim() {
    if (!currentData || manualReplayOrders.length === 0) {
        manualSimResult = null;
        document.getElementById('btn-close-manual').style.display = 'none';
        return;
    }
    const maxIdx = replayActive ? replayIndex : (currentData.candles.length - 1);
    const candles = currentData.candles;
    const initCap = parseFloat(document.getElementById('initial_capital').value) || 100000;
    const enableLadder = document.getElementById('enable_scale_out').checked;
    const t1GainPct = parseFloat(document.getElementById('t1_gain_pct').value) || 4.0;
    const t1SellPct = parseFloat(document.getElementById('t1_sell_pct').value) || 35.0;
    const t2GainPct = parseFloat(document.getElementById('t2_gain_pct').value) || 8.0;
    const t2SellPct = parseFloat(document.getElementById('t2_sell_pct').value) || 35.0;
    const isNoTp = (document.getElementById('strategy').value === 'no_tp_trend');
    const useOriginal = (document.getElementById('engine_mode').value === 'original');

    let cash = initCap;
    let trades = [];
    let markers = [];
    let boxes = [];
    let hasOpenPos = false;

    for (let oIdx = 0; oIdx < manualReplayOrders.length; oIdx++) {
        const ord = manualReplayOrders[oIdx];
        if (ord.entryIdx > maxIdx) continue;
        const entryCandle = candles[ord.entryIdx];
        const bInfo = currentData.bar_details[String(entryCandle.time)] || {};
        const ep = entryCandle.close;
        const isLong = (ord.side === 'LONG');
        const unitVol = bInfo.unit_vol || bInfo.atr || (ep * 0.015);
        const dSl = bInfo.dyn_sl_mult || parseFloat(document.getElementById('sl_mult').value) || 1.5;
        const dTp = isNoTp ? 999.0 : (bInfo.dyn_tp_mult || parseFloat(document.getElementById('tp_mult').value) || 2.8);

        let slPrice = isLong ? (ep - dSl * unitVol) : (ep + dSl * unitVol);
        const initSlPrice = slPrice;
        const tpPrice = isLong ? (ep + dTp * unitVol) : (ep - dTp * unitVol);

        const fullDist = Math.abs(tpPrice - ep);
        const pctDist1 = ep * (Math.max(t1GainPct, 0.1) / 100.0);
        const pctDist2 = ep * (Math.max(t2GainPct, 0.2) / 100.0);
        const d1 = Math.max(Math.min(pctDist1, fullDist * 0.5), ep * 0.002);
        const d2 = Math.max(Math.min(pctDist2, fullDist * 0.8), d1 * 1.35);
        const t1Target = isLong ? (ep + d1) : (ep - d1);
        const t2Target = isLong ? (ep + d2) : (ep - d2);

        const targetDollars = cash * (ord.allocPct / 100.0);
        const initShares = Math.max(1, Math.floor(targetDollars / Math.max(ep, 0.01)));
        let remShares = initShares;
        const posVal = initShares * ep;
        const dollarRisk = initShares * Math.abs(ep - initSlPrice);

        let realizedPnl = 0.0;
        let scaleStage = 0;
        let scaleNotes = [];
        let closed = false;
        let exitBarIdx = maxIdx;
        let exitPrice = candles[maxIdx].close;
        let resType = 'OPEN';
        let runningExtreme = isLong ? entryCandle.high : entryCandle.low;

        markers.push({
            time: entryCandle.time, position: isLong ? 'belowBar' : 'aboveBar',
            color: isLong ? '#00E676' : '#FF4081', shape: isLong ? 'arrowUp' : 'arrowDown',
            text: `MANUAL ${ord.side} ${initShares}sh (${ord.allocPct}% Eq @ $${ep.toFixed(2)})`
        });

        for (let j = ord.entryIdx + 1; j <= maxIdx; j++) {
            const c = candles[j];
            if (ord.forceCloseIdx !== null && j >= ord.forceCloseIdx) {
                exitPrice = c.close;
                const remPnl = (isLong ? (exitPrice - ep) : (ep - exitPrice)) * remShares;
                realizedPnl += remPnl;
                remShares = 0;
                closed = true;
                exitBarIdx = j;
                resType = realizedPnl >= 0 ? 'MANUAL-WIN' : 'MANUAL-EXIT';
                markers.push({
                    time: c.time, position: isLong ? 'aboveBar' : 'belowBar', color: '#FFD600',
                    shape: isLong ? 'arrowDown' : 'arrowUp', text: `MANUAL CLOSE (${realizedPnl >= 0 ? '+' : ''}$${Math.round(realizedPnl).toLocaleString()})`
                });
                break;
            }

            if (!useOriginal) {
                if (isLong) {
                    if (c.high > runningExtreme) runningExtreme = c.high;
                    const potentialSl = runningExtreme - (dSl * unitVol);
                    if (potentialSl > slPrice) slPrice = potentialSl;
                } else {
                    if (c.low < runningExtreme) runningExtreme = c.low;
                    const potentialSl = runningExtreme + (dSl * unitVol);
                    if (potentialSl < slPrice) slPrice = potentialSl;
                }
            }

            const hitSlFirst = isLong ? (c.low <= slPrice) : (c.high >= slPrice);
            if (isNoTp) {
                const curBInfo = currentData.bar_details[String(c.time)] || {};
                const potentialNewSl = isLong ? (curBInfo.ema20 - dSl * unitVol) : (curBInfo.ema20 + dSl * unitVol);
                if (isLong && potentialNewSl > slPrice) slPrice = potentialNewSl;
                if (!isLong && potentialNewSl < slPrice) slPrice = potentialNewSl;
            }

            if (enableLadder && !isNoTp && !useOriginal && !hitSlFirst && remShares >= 3) {
                if (scaleStage === 0 && (isLong ? (c.high >= t1Target && t1Target < tpPrice) : (c.low <= t1Target && t1Target > tpPrice))) {
                    const qty1 = Math.min(remShares - 1, Math.max(1, Math.round(initShares * (t1SellPct / 100.0))));
                    const p1 = (isLong ? (t1Target - ep) : (ep - t1Target)) * qty1;
                    realizedPnl += p1;
                    remShares -= qty1;
                    scaleStage = 1;
                    slPrice = isLong ? Math.max(slPrice, ep) : Math.min(slPrice, ep);
                    scaleNotes.push("T1(-" + qty1 + "sh)");
                }
            }

            const hitTp = (c.high >= tpPrice) && !isNoTp;
            const hitSl = isLong ? (c.low <= slPrice) : (c.high >= slPrice);

            if (hitTp || hitSl) {
                exitPrice = hitTp ? tpPrice : slPrice;
                const remPnl = (isLong ? (exitPrice - ep) : (ep - exitPrice)) * remShares;
                realizedPnl += remPnl;
                remShares = 0;
                closed = true;
                exitBarIdx = j;
                resType = hitTp ? 'TP' : (realizedPnl > 0 ? 'TRAIL-WIN' : 'SL');
                markers.push({
                    time: c.time, position: isLong ? 'aboveBar' : 'belowBar',
                    color: (hitTp || realizedPnl > 0) ? '#FFD600' : '#FF1744',
                    shape: isLong ? 'arrowDown' : 'arrowUp',
                    text: `${isLong ? 'L' : 'S'}-${resType} (${realizedPnl >= 0 ? '+' : ''}$${Math.round(realizedPnl).toLocaleString()})`
                });
                break;
            }
        }

        let totalTradePnl = realizedPnl;
        if (!closed) {
            hasOpenPos = true;
            exitPrice = candles[maxIdx].close;
            const unreal = (isLong ? (exitPrice - ep) : (ep - exitPrice)) * remShares;
            totalTradePnl += unreal;
        } else {
            cash += totalTradePnl;
        }

        const eqAfter = closed ? cash : (cash + totalTradePnl);
        const retPct = (totalTradePnl / Math.max(posVal, 1)) * 100.0;
        const exitCandle = candles[exitBarIdx];
        const exitInfo = currentData.bar_details[String(exitCandle.time)] || {};

        trades.push({
            trade_no: 100 + oIdx + 1, side: ord.side, signal_type: `MANUAL REPLAY (${ord.allocPct}% Eq)`,
            pool: `Manual (${ord.allocPct}% Portfolio)`, entry_bar_idx: ord.entryIdx, exit_bar_idx: exitBarIdx,
            entry_time: entryCandle.time, entry_dt: bInfo.dt || String(entryCandle.time),
            exit_time: exitCandle.time, exit_dt: closed ? (exitInfo.dt || String(exitCandle.time)) : 'OPEN (Replay Active)',
            type: resType, entry_price: Number(ep.toFixed(2)), exit_price: Number(exitPrice.toFixed(2)),
            sl_price: Number(initSlPrice.toFixed(2)), tp_price: Number(tpPrice.toFixed(2)),
            dyn_mults: `${dSl.toFixed(2)}x`, scale_summary: scaleNotes.length > 0 ? scaleNotes.join(' -> ') : (closed ? '100% @ Exit' : `OPEN (${remShares}sh)`),
            shares: initShares, position_value: Number(posVal.toFixed(2)), alloc_pct: Number(ord.allocPct.toFixed(1)),
            dollar_risk: Number(dollarRisk.toFixed(2)), bar_volume: bInfo.vol || 0, bar_rvol: bInfo.rvol || 1.0,
            pnl: Number(totalTradePnl.toFixed(2)), ret_pct: Number(retPct.toFixed(2)), equity_after: Number(eqAfter.toFixed(2))
        });
        boxes.push({
            trade_no: 100 + oIdx + 1, side: ord.side, t1: entryCandle.time, t2: exitCandle.time,
            entry: Number(ep.toFixed(2)), tp: Number(tpPrice.toFixed(2)), sl: Number(initSlPrice.toFixed(2)),
            t1_lvl: Number(t1Target.toFixed(2)), t2_lvl: Number(t2Target.toFixed(2)), outcome: closed ? resType : 'ACTIVE'
        });
    }

    document.getElementById('btn-close-manual').style.display = hasOpenPos ? 'inline-block' : 'none';
    const latestEq = trades.length > 0 ? trades[trades.length - 1].equity_after : initCap;
    manualSimResult = {
        trades, markers, boxes, hasOpenPosition: hasOpenPos,
        endingEquity: latestEq, netReturnPct: Number((((latestEq - initCap) / initCap) * 100).toFixed(2))
    };

    // Update Live Trade Inspector Drawer
    updateLiveTradeInspector(candles, maxIdx);
}

function updateLiveTradeInspector(candles, maxIdx) {
    const drawer = document.getElementById('inspector-drawer');
    const statusEl = document.getElementById('inspector-status');
    if (!replayActive || !manualSimResult || !manualSimResult.hasOpenPosition) {
        if (drawer) drawer.style.display = 'none';
        return;
    }

    const activeOrders = manualSimResult.trades.filter(t => t.entry_bar_idx <= maxIdx && maxIdx <= t.exit_bar_idx);
    if (activeOrders.length === 0) {
        drawer.style.display = 'none';
        return;
    }

    drawer.style.display = 'block';
    const trade = activeOrders[activeOrders.length - 1];
    statusEl.textContent = `${trade.side} @ $${trade.entry_price} | SL: $${trade.sl_price} | PnL: ${trade.ret_pct >= 0 ? '+' : ''}${trade.ret_pct}%`;

    const subCandles = candles.slice(trade.entry_bar_idx, maxIdx + 1);
    inspectorCandleSeries.setData(subCandles);
    inspectorChart.timeScale().fitContent();
}

function renderReplayFrame(resetViewport = false) {
    if (!currentData || !currentData.candles || currentData.candles.length === 0) return;
    const maxIdx = replayActive ? replayIndex : (currentData.candles.length - 1);
    const cutoffTime = currentData.candles[maxIdx].time;
    const visCandles = currentData.candles.slice(0, maxIdx + 1);
    const visVolumes = currentData.volumes.slice(0, maxIdx + 1);

    candleSeries.setData(visCandles);
    volumeSeries.setData(visVolumes);

    recomputeManualReplaySim();

    const stratMarkers = (currentData.backtest.markers || []).filter(m => m.time <= cutoffTime);
    const manMarkers = manualSimResult ? manualSimResult.markers : [];
    const combinedMarkers = [...stratMarkers, ...manMarkers].sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(combinedMarkers);

    const curBarInfo = currentData.bar_details[String(cutoffTime)] || {};
    const statusEl = document.getElementById('replay-status-text');
    if (replayActive) {
        const eqStr = manualSimResult ? ` | Manual Replay Equity: $${manualSimResult.endingEquity.toLocaleString()} (${manualSimResult.netReturnPct >= 0 ? '+' : ''}${manualSimResult.netReturnPct}%)` : '';
        statusEl.textContent = `Bar ${maxIdx + 1} / ${currentData.candles.length} — ${curBarInfo.dt || ''} | Close: $${curBarInfo.c || ''}${eqStr}`;
    } else {
        statusEl.textContent = '[Full History Mode]';
    }

    renderOverlays();
    updateTradeLogTable(currentData);

    if (resetViewport) {
        chart.timeScale().fitContent();
    }
}

function focusTradeOnChart(t, rowEl) {
    if (!chart || !currentData) return;
    focusedTradeNo = t.trade_no;
    document.querySelectorAll('#trade-log-table tbody tr').forEach(r => r.classList.remove('selected-trade-row'));
    if (rowEl) rowEl.classList.add('selected-trade-row');

    const eIdx = (typeof t.entry_bar_idx === 'number') ? t.entry_bar_idx : 0;
    const xIdx = (typeof t.exit_bar_idx === 'number') ? t.exit_bar_idx : eIdx + 1;
    const spanBars = Math.max(1, xIdx - eIdx);
    const padBars = Math.max(18, Math.round(spanBars * 1.35));
    const fromLogical = Math.max(-5, eIdx - padBars);
    const toLogical = Math.min(currentData.total_bars + 5, xIdx + padBars);

    chart.priceScale('right').applyOptions({ autoScale: true });
    chart.timeScale().setVisibleLogicalRange({ from: fromLogical, to: toLogical });

    const boxChk = document.getElementById('chk_tpsl_boxes');
    if (boxChk && !boxChk.checked) {
        boxChk.checked = true;
    }
    requestAnimationFrame(drawCanvasZones);
}

function renderBreadcrumbs() {
    const bar = document.getElementById('breadcrumb-bar');
    bar.innerHTML = '';
    breadcrumbs.forEach((b, idx) => {
        const el = document.createElement('span');
        el.className = 'crumb' + (idx === activeCrumbIdx ? ' active' : '');
        const lbl = document.createElement('span');
        lbl.textContent = b.label;
        lbl.onclick = () => {
            activeCrumbIdx = idx;
            currentTF = b.tf;
            currentStartTs = b.startTs;
            asyncActionWrapper(loadChartData);
        };
        el.appendChild(lbl);

        const closeBtn = document.createElement('span');
        closeBtn.className = 'crumb-close';
        closeBtn.textContent = '×';
        closeBtn.title = 'Delete this navigation tab';
        closeBtn.onclick = (e) => {
            e.stopPropagation();
            deleteCrumb(idx);
        };
        el.appendChild(closeBtn);
        bar.appendChild(el);
    });

    if (breadcrumbs.length > 1) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'crumb-clear';
        clearBtn.textContent = 'Clear All';
        clearBtn.onclick = () => asyncActionWrapper(resetToRoot);
        bar.appendChild(clearBtn);
    }

    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase() === currentTF.toLowerCase());
    });
}

function deleteCrumb(idx) {
    if (breadcrumbs.length <= 1) {
        asyncActionWrapper(resetToRoot);
        return;
    }
    breadcrumbs.splice(idx, 1);
    if (activeCrumbIdx >= breadcrumbs.length) {
        activeCrumbIdx = breadcrumbs.length - 1;
    }
    const activeTab = breadcrumbs[activeCrumbIdx];
    currentTF = activeTab.tf;
    currentStartTs = activeTab.startTs;
    asyncActionWrapper(loadChartData);
}

function switchTF(tf) {
    currentTF = tf;
    if (activeCrumbIdx === 0) {
        currentStartTs = null;
    }
    const datePrefix = currentStartTs ? new Date(currentStartTs * 1000).toISOString().slice(0, 10) : 'Max';
    breadcrumbs[activeCrumbIdx] = { label: `${datePrefix} (${tf})`, tf: tf, startTs: currentStartTs };
    return loadChartData();
}

function resetToRoot() {
    if (replayTimer) toggleReplayAutoPlay();
    replayActive = false;
    manualReplayOrders = [];
    manualSimResult = null;
    currentTF = '1Y';
    currentStartTs = null;
    currentCutoffTs = null;
    manualEntryTs = null;
    focusedTradeNo = null;
    breadcrumbs = [{ label: `Max (1Y)`, tf: '1Y', startTs: null }];
    activeCrumbIdx = 0;
    return loadChartData();
}

function clearCutoffAndManual() {
    if (replayTimer) toggleReplayAutoPlay();
    replayActive = false;
    manualReplayOrders = [];
    manualSimResult = null;
    currentCutoffTs = null;
    manualEntryTs = null;
    focusedTradeNo = null;
    return loadChartData();
}

async function loadChartData(isBackgroundRefresh = false) {
    renderBreadcrumbs();
    if (!isBackgroundRefresh) focusedTradeNo = null;

    const params = new URLSearchParams({
        ticker: document.getElementById('ticker').value,
        tf: currentTF,
        strategy: document.getElementById('strategy').value,
        trade_side: document.getElementById('trade_side').value,
        engine_mode: document.getElementById('engine_mode').value,
        ema200_filter: document.getElementById('ema200_filter').value,
        sl_mult: document.getElementById('sl_mult').value,
        tp_mult: document.getElementById('tp_mult').value,
        dynamic_tpsl: document.getElementById('dynamic_tpsl').value,
        vol_model: document.getElementById('vol_model').value,
        enable_scale_out: document.getElementById('enable_scale_out').checked ? 'true' : 'false',
        t1_gain_pct: document.getElementById('t1_gain_pct').value,
        t1_sell_pct: document.getElementById('t1_sell_pct').value,
        t2_gain_pct: document.getElementById('t2_gain_pct').value,
        t2_sell_pct: document.getElementById('t2_sell_pct').value,
        w_trend: document.getElementById('w_trend').value,
        w_liq: document.getElementById('w_liq').value,
        w_rev: document.getElementById('w_rev').value,
        initial_capital: document.getElementById('initial_capital').value,
        core_pct: document.getElementById('core_pct').value,
        intraday_pct: document.getElementById('intraday_pct').value,
        risk_pct: document.getElementById('risk_pct').value,
        manual_side: manualSide,
        manual_alloc_pct: document.getElementById('manual_alloc_pct').value,
        force_refresh: isBackgroundRefresh ? 'true' : 'false'
    });

    if (currentStartTs) params.append('start_ts', currentStartTs);
    if (currentCutoffTs) params.append('cutoff_ts', currentCutoffTs);
    if (manualEntryTs) params.append('manual_entry_ts', manualEntryTs);

    const res = await fetch(`/api/analyze?${params.toString()}`);
    const data = await res.json();
    const banner = document.getElementById('alert-banner');

    if (data.error) {
        banner.style.display = 'block';
        banner.textContent = data.error;
        return;
    }

    currentData = data;
    let msgs = [];
    if (data.warning) msgs.push(data.warning);
    if (data.unavailable_indicators && data.unavailable_indicators.length > 0) {
        msgs.push("Note: " + data.unavailable_indicators.join(", "));
    }
    if (msgs.length > 0) {
        banner.style.display = 'block';
        banner.textContent = msgs.join(" | ");
    } else {
        banner.style.display = 'none';
    }

    if (!isBackgroundRefresh) {
        const allIndIds = ['ema20', 'ema200', 'vwap', 'vwap_bands', 'atr_bands', 'gk_bands', 'fib_levels', 'orb_levels', 'fvg_zones'];
        allIndIds.forEach(id => {
            const el = document.getElementById('chk_' + id);
            if (el) el.checked = data.auto_indicators.includes(id);
        });
    }

    if (!replayActive) {
        replayIndex = Math.max(0, data.candles.length - 1);
    } else {
        replayIndex = Math.min(replayIndex, Math.max(0, data.candles.length - 1));
    }

    renderReplayFrame(!isBackgroundRefresh);
    updateSummaryTable(data);

    document.getElementById('opt_spot').value = data.latest_metrics.close;
    if (!isBackgroundRefresh) {
        document.getElementById('opt_strike').value = Math.round(data.latest_metrics.close * 1.05);
    }
    document.getElementById('opt_gk').value = data.latest_metrics.gk_daily_vol_pct;
    calcOptions();
    await loadOptionChain(document.getElementById('chain_expiry').value || '');
    setTimeout(resizeChart, 50);
}

function renderOverlays() {
    if (!currentData) return;
    const s = currentData.series;
    const maxIdx = replayActive ? replayIndex : (currentData.candles.length - 1);
    const cutoffTime = currentData.candles[maxIdx] ? currentData.candles[maxIdx].time : Infinity;
    const sliceToReplay = (arr) => (arr || []).filter(pt => pt.time <= cutoffTime);

    const showEma = document.getElementById('chk_ema20').checked;
    lineSeriesMap.ema20.setData(showEma ? sliceToReplay(s.ema20) : []);
    lineSeriesMap.ema50.setData(showEma ? sliceToReplay(s.ema50) : []);

    const showEma200 = document.getElementById('chk_ema200').checked;
    lineSeriesMap.ema200.setData(showEma200 ? sliceToReplay(s.ema200) : []);

    const showVwap = document.getElementById('chk_vwap').checked;
    lineSeriesMap.vwap.setData(showVwap ? sliceToReplay(s.vwap) : []);

    const showVwapBands = document.getElementById('chk_vwap_bands').checked;
    lineSeriesMap.vwap_u2.setData(showVwapBands ? sliceToReplay(s.vwap_u2) : []);
    lineSeriesMap.vwap_l2.setData(showVwapBands ? sliceToReplay(s.vwap_l2) : []);

    const showAtr = document.getElementById('chk_atr_bands').checked;
    lineSeriesMap.atr_upper.setData(showAtr ? sliceToReplay(s.atr_upper) : []);
    lineSeriesMap.atr_lower.setData(showAtr ? sliceToReplay(s.atr_lower) : []);

    const showGk = document.getElementById('chk_gk_bands').checked;
    lineSeriesMap.gk_upper.setData(showGk ? sliceToReplay(s.gk_upper) : []);
    lineSeriesMap.gk_lower.setData(showGk ? sliceToReplay(s.gk_lower) : []);

    const showFib = document.getElementById('chk_fib_levels').checked;
    lineSeriesMap.fib_382.setData(showFib ? sliceToReplay(s.fib_382) : []);
    lineSeriesMap.fib_500.setData(showFib ? sliceToReplay(s.fib_500) : []);
    lineSeriesMap.fib_618.setData(showFib ? sliceToReplay(s.fib_618) : []);
    lineSeriesMap.fib_786.setData(showFib ? sliceToReplay(s.fib_786) : []);

    const showOrb = document.getElementById('chk_orb_levels').checked;
    lineSeriesMap.orb_high.setData(showOrb ? sliceToReplay(s.orb_high) : []);
    lineSeriesMap.orb_low.setData(showOrb ? sliceToReplay(s.orb_low) : []);

    const st = currentData.backtest.stats;
    const lm = currentData.latest_metrics;
    const filterMode = document.getElementById('ema200_filter').selectedOptions[0].text;
    const engineMode = document.getElementById('engine_mode').selectedOptions[0].text;

    document.getElementById('chart-legend').innerHTML = `
        <b>${currentData.ticker} [${currentData.tf}]</b> (${currentData.first_date} to ${currentData.last_date} | ${maxIdx + 1}/${currentData.total_bars} bars | Server: ${currentData.server_updated_at})<br>
        Close: <b>$${lm.close}</b> | <span style="color:#ffd600">EMA 200: $${lm.ema200}</span> | ATR: $${lm.atr} | GK Vol: ${lm.gk_daily_vol_pct}%<br>
        <span style="color:#00e5ff">Engine: ${engineMode}</span> | <span style="color:#00e5ff">Filter: ${filterMode}</span><br>
        <span style="color:#00e676">Strategy Equity: $${st.ending_equity.toLocaleString()} (${st.net_return_pct >= 0 ? '+' : ''}${st.net_return_pct}%)</span> | Trades: ${st.total_trades} (${st.long_trades}L/${st.short_trades}S) | Win Rate: ${st.win_rate_pct}%
    `;
    requestAnimationFrame(drawCanvasZones);
}

function drawCanvasZones() {
    const canvas = document.getElementById('overlay-canvas');
    const wrap = document.getElementById('chart-wrap');
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!currentData) return;

    const tsScale = chart.timeScale();
    const maxIdx = replayActive ? replayIndex : (currentData.candles.length - 1);
    const cutoffTime = currentData.candles[maxIdx] ? currentData.candles[maxIdx].time : Infinity;

    if (document.getElementById('chk_unclear').checked && currentData.unclear_spans) {
        ctx.fillStyle = 'rgba(120, 123, 134, 0.14)';
        ctx.font = '10px sans-serif';
        currentData.unclear_spans.forEach(span => {
            if (span.t1 > cutoffTime) return;
            const x1 = tsScale.timeToCoordinate(span.t1);
            const x2 = tsScale.timeToCoordinate(Math.min(span.t2, cutoffTime));
            if (x1 !== null && x2 !== null && (x2 - x1) > 12) {
                ctx.fillRect(x1, 0, x2 - x1, canvas.height - 28);
                ctx.fillStyle = 'rgba(180, 185, 195, 0.55)';
                ctx.fillText('VOLATILE / UNCLEAR', x1 + 4, 24);
                ctx.fillStyle = 'rgba(120, 123, 134, 0.14)';
            }
        });
    }

    if (document.getElementById('chk_fvg_zones').checked && currentData.fvg_boxes) {
        currentData.fvg_boxes.forEach(b => {
            if (b.t1 > cutoffTime) return;
            const x1 = tsScale.timeToCoordinate(b.t1);
            const x2 = tsScale.timeToCoordinate(Math.min(b.t2, cutoffTime));
            const yTop = candleSeries.priceToCoordinate(b.top);
            const yBot = candleSeries.priceToCoordinate(b.bot);
            if (x1 === null || x2 === null || yTop === null || yBot === null) return;
            const w = Math.max(x2 - x1, 8);
            const h = Math.abs(yBot - yTop);
            ctx.fillStyle = b.kind === 'BULL_FVG' ? 'rgba(0, 176, 255, 0.18)' : 'rgba(255, 152, 0, 0.18)';
            ctx.strokeStyle = b.kind === 'BULL_FVG' ? 'rgba(0, 176, 255, 0.6)' : 'rgba(255, 152, 0, 0.6)';
            ctx.fillRect(x1, Math.min(yTop, yBot), w, Math.max(h, 2));
            ctx.strokeRect(x1, Math.min(yTop, yBot), w, Math.max(h, 2));
        });
    }

    if (document.getElementById('chk_tpsl_boxes').checked) {
        const stratBoxes = (currentData.backtest.boxes || []).filter(b => b.t1 <= cutoffTime);
        const manBoxes = manualSimResult ? manualSimResult.boxes : [];
        const allBoxes = [...stratBoxes, ...manBoxes];

        allBoxes.forEach(b => {
            const x1 = tsScale.timeToCoordinate(b.t1);
            const x2 = tsScale.timeToCoordinate(Math.min(b.t2, cutoffTime));
            const yEntry = candleSeries.priceToCoordinate(b.entry);
            const yTp = candleSeries.priceToCoordinate(b.tp);
            const ySl = candleSeries.priceToCoordinate(b.sl);
            if (x1 === null || x2 === null || yEntry === null || yTp === null || ySl === null) return;

            const w = Math.max(x2 - x1, 14);
            const isFocused = (focusedTradeNo !== null && b.trade_no === focusedTradeNo);

            ctx.fillStyle = isFocused ? 'rgba(0, 230, 118, 0.28)' : 'rgba(0, 230, 118, 0.15)';
            ctx.strokeStyle = isFocused ? '#00e676' : 'rgba(0, 230, 118, 0.75)';
            ctx.lineWidth = isFocused ? 2 : 1;
            ctx.fillRect(x1, Math.min(yTp, yEntry), w, Math.abs(yEntry - yTp));
            ctx.strokeRect(x1, Math.min(yTp, yEntry), w, Math.abs(yEntry - yTp));

            ctx.fillStyle = isFocused ? 'rgba(255, 23, 68, 0.28)' : 'rgba(255, 23, 68, 0.15)';
            ctx.strokeStyle = isFocused ? '#ff1744' : 'rgba(255, 23, 68, 0.75)';
            ctx.fillRect(x1, Math.min(yEntry, ySl), w, Math.abs(ySl - yEntry));
            ctx.strokeRect(x1, Math.min(yEntry, ySl), w, Math.abs(ySl - yEntry));

            if (isFocused) {
                const topY = Math.min(yTp, ySl, yEntry) - 6;
                const botY = Math.max(yTp, ySl, yEntry) + 6;
                ctx.save();
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 2.5;
                ctx.strokeRect(x1 - 4, topY, w + 8, botY - topY);
                ctx.fillStyle = '#00e5ff';
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(`Trade #${b.trade_no} (${b.side})`, x1 - 2, Math.max(14, topY - 4));
                ctx.restore();
            }
        });
    }
}

function updateTradeLogTable(data) {
    const tbody = document.querySelector('#trade-log-table tbody');
    tbody.innerHTML = '';
    const maxIdx = replayActive ? replayIndex : (data.candles.length - 1);
    const cutoffTime = data.candles[maxIdx] ? data.candles[maxIdx].time : Infinity;
    const stratTrades = (data.backtest.trades || []).filter(t => t.entry_time <= cutoffTime);
    const manTrades = manualSimResult ? manualSimResult.trades : [];
    const trades = [...manTrades, ...stratTrades];

    document.getElementById('trade-count-badge').textContent = `${trades.length} Executions (${manTrades.length} Manual Replay)`;
    if (trades.length === 0) {
        tbody.innerHTML = `<tr><td colspan="17" style="text-align:center; color:#787b86; padding:12px;">No trades triggered on this timeframe/view yet. Use Bar-by-Bar Replay or switch strategy.</td></tr>`;
        return;
    }

    trades.forEach(t => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.title = 'Click to zoom chart directly to this trade';
        tr.onclick = () => focusTradeOnChart(t, tr);

        const sideCls = t.side === 'LONG' ? 'pos' : 'neg';
        const pnlCls = t.pnl >= 0 ? 'pos' : 'neg';
        const resBadge = (['TP', 'TRAIL-WIN', 'MANUAL-WIN'].includes(t.type)) ? `<span class="pill pill-cheap">${t.type}</span>` : (t.type === 'SL' ? `<span class="pill pill-exp">SL LOSS</span>` : `<span class="pill pill-fair">${t.type}</span>`);

        tr.innerHTML = `
            <td>${t.trade_no}</td>
            <td class="${sideCls}"><b>${t.side}</b></td>
            <td>${t.signal_type}</td>
            <td>${t.pool}</td>
            <td>${t.entry_dt}</td>
            <td><b>$${t.entry_price}</b></td>
            <td style="color:#ffd600; font-weight:700;">${t.shares.toLocaleString()} sh (RVOL ${t.bar_rvol}x)</td>
            <td>$${t.position_value.toLocaleString()} (${t.alloc_pct}%)</td>
            <td>$${t.dollar_risk.toLocaleString()}</td>
            <td>$${t.sl_price} / ${t.tp_price > 0 ? '$' + t.tp_price : 'No TP (Trailing)'}</td>
            <td style="color:#80d8ff; font-size:10.5px;">${t.scale_summary || '100% @ Exit'}</td>
            <td>${t.exit_dt}</td>
            <td>$${t.exit_price}</td>
            <td>${resBadge}</td>
            <td class="${pnlCls}"><b>${t.pnl >= 0 ? '+' : ''}$${t.pnl.toLocaleString()}</b></td>
            <td class="${pnlCls}">${t.ret_pct >= 0 ? '+' : ''}${t.ret_pct}%</td>
            <td><b>$${t.equity_after.toLocaleString()}</b></td>
        `;
        tbody.appendChild(tr);
    });
}

function updateSummaryTable(data) {
    const tbody = document.querySelector('#summary-table tbody');
    tbody.innerHTML = '';
    const activeKey = document.getElementById('strategy').value;

    if (data.all_strategies_summary.length > 0) {
        const bh = data.all_strategies_summary[0].buy_hold_pct;
        document.getElementById('bh-badge').textContent = `Buy & Hold: ${bh >= 0 ? '+' : ''}${bh}%`;
    }

    data.all_strategies_summary.forEach(row => {
        const tr = document.createElement('tr');
        if (row.strategy_key === activeKey) tr.className = 'highlight-row';
        tr.style.cursor = 'pointer';
        tr.onclick = () => {
            document.getElementById('strategy').value = row.strategy_key;
            onStrategyChange();
        };

        const cls = row.net_return_pct >= 0 ? 'pos' : 'neg';
        tr.innerHTML = `
            <td>${row.strategy_name}</td>
            <td>${row.bucket}</td>
            <td>$${row.ending_equity.toLocaleString()}</td>
            <td class="${cls}">${row.net_return_pct >= 0 ? '+' : ''}${row.net_return_pct}%</td>
            <td>${row.total_trades} (${row.long_trades}L / ${row.short_trades}S)</td>
            <td>${row.win_rate_pct}%</td>
            <td>${row.profit_factor}</td>
            <td class="neg">-${row.max_dd_pct}%</td>
        `;
        tbody.appendChild(tr);
    });
}

async function calcOptions() {
    const params = new URLSearchParams({
        spot: document.getElementById('opt_spot').value,
        strike: document.getElementById('opt_strike').value,
        dte: document.getElementById('opt_dte').value,
        iv_pct: document.getElementById('opt_iv').value,
        gk_daily_vol_pct: document.getElementById('opt_gk').value,
        opt_type: document.getElementById('opt_type').value
    });
    const res = await fetch(`/api/options_greeks?${params.toString()}`);
    const d = await res.json();
    document.getElementById('opt-results').innerHTML = `
        <div><b>Theoretical Value:</b> $${d.theo_price} | <b>Breakeven:</b> $${d.breakeven} | <b>Prob ITM:</b> ${d.prob_itm_pct}%</div>
        <div><b>Greeks:</b> Delta: ${d.delta} | Gamma: ${d.gamma} | Theta: $${d.theta_daily}/day | Vega: $${d.vega_per_1pct}</div>
        <div><b>Expected Move (+-1s):</b> Implied: +-$${d.expected_move_iv} vs. GK Realized: +-$${d.expected_move_hv} (IV/HV: ${d.iv_hv_ratio}x)</div>
        <div style="margin-top:3px; color:#ffd600;"><b>Verdict:</b> ${d.verdict}</div>
    `;
}

async function loadOptionChain(selectedExp = '') {
    if (!currentData) return;
    const params = new URLSearchParams({
        ticker: currentData.ticker,
        spot: currentData.latest_metrics.close,
        gk_daily_vol_pct: currentData.latest_metrics.gk_daily_vol_pct,
        max_dte: 60
    });
    if (selectedExp) params.append('expiry', selectedExp);

    const res = await fetch(`/api/option_chain?${params.toString()}`);
    const data = await res.json();
    document.getElementById('chain-source').textContent = data.source;

    const expSel = document.getElementById('chain_expiry');
    expSel.innerHTML = '';
    (data.expirations || []).forEach(exp => {
        const opt = document.createElement('option');
        opt.value = exp;
        opt.textContent = `${exp}`;
        if (exp === data.selected_expiry) opt.selected = true;
        expSel.appendChild(opt);
    });

    if (data.dte) {
        document.getElementById('opt_dte').value = data.dte;
    }

    const renderSide = (rows, tbodyId, optType) => {
        const tb = document.querySelector(`#${tbodyId} tbody`);
        tb.innerHTML = '';
        (rows || []).forEach(r => {
            const tr = document.createElement('tr');
            tr.style.cursor = 'pointer';
            tr.onclick = () => {
                document.getElementById('opt_strike').value = r.strike;
                document.getElementById('opt_iv').value = r.iv_pct;
                document.getElementById('opt_type').value = optType;
                if (r.dte) document.getElementById('opt_dte').value = r.dte;
                calcOptions();
                lastSelectedOptionContract = { sym: r.contractSymbol, strike: r.strike, optType, ivPct: r.iv_pct };
                asyncActionWrapper(() => loadContractHistory(r.contractSymbol, r.strike, optType, r.iv_pct));
            };
            const pillCls = r.verdict === 'CHEAP' ? 'pill-cheap' : (r.verdict === 'EXPENSIVE' ? 'pill-exp' : 'pill-fair');
            tr.innerHTML = `
                <td><b>$${r.strike}</b></td>
                <td>${r.dte || data.dte || '-'}d</td>
                <td>$${r.lastPrice}</td>
                <td>$${r.bid} / $${r.ask}</td>
                <td>${r.iv_pct}%</td>
                <td>${r.delta}</td>
                <td>${r.volume} / ${r.openInterest}</td>
                <td><span class="pill ${pillCls}">${r.verdict}</span></td>
            `;
            tb.appendChild(tr);
        });
    };

    renderSide(data.calls, 'calls-table', 'call');
    renderSide(data.puts, 'puts-table', 'put');

    if (lastSelectedOptionContract) {
        await loadContractHistory(
            lastSelectedOptionContract.sym,
            lastSelectedOptionContract.strike,
            lastSelectedOptionContract.optType,
            lastSelectedOptionContract.ivPct
        );
    }
}

async function loadContractHistory(contractSym, strike, optType, ivPct) {
    const params = new URLSearchParams({
        contract_symbol: contractSym,
        ticker: currentData.ticker,
        strike: strike,
        opt_type: optType,
        iv_pct: ivPct
    });
    const res = await fetch(`/api/option_contract_history?${params.toString()}`);
    const d = await res.json();
    const box = document.getElementById('contract-history-box');
    box.style.display = 'block';
    document.getElementById('contract-hist-title').textContent = `Option Contract 60-Day History: ${d.contract} (${d.source}) — Showing All ${d.bars.length} Sessions`;

    const strip = document.getElementById('contract-hist-strip');
    strip.innerHTML = '';
    d.bars.slice(-60).forEach(b => {
        const card = document.createElement('div');
        card.style.cssText = 'background:#131722; border:1px solid #2a2e39; border-radius:4px; padding:5px 8px; min-width:92px; font-size:10.5px;';
        card.innerHTML = `<div style="color:#787b86">${b.date}</div><div style="font-weight:700; color:#00e676">$${b.close}</div><div style="color:#90a4ae">H:$${b.high} L:$${b.low}</div>`;
        strip.appendChild(card);
    });
}

function startOneMinuteAutoRefreshLoop() {
    setInterval(async () => {
        liveRefreshCountdown--;
        const badge = document.getElementById('live-refresh-badge');
        if (liveRefreshCountdown <= 0) {
            liveRefreshCountdown = 60;
            if (badge) badge.textContent = '⟳ Syncing 1m & Options...';
            if (currentData && !replayActive) {
                if (currentTF === '1m') {
                    await loadChartData(true);
                } else {
                    await loadOptionChain(document.getElementById('chain_expiry').value || '');
                }
            } else if (currentData && replayActive) {
                await loadOptionChain(document.getElementById('chain_expiry').value || '');
            }
        }
        if (badge) {
            badge.textContent = `⟳ 1m & Options Sync: ${liveRefreshCountdown}s`;
        }
    }, 1000);
}

window.onload = () => {
    initChart();
    onStrategyChange();
    asyncActionWrapper(resetToRoot);
    startOneMinuteAutoRefreshLoop();
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
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

app = FastAPI(title="Quant Lab: Multi-Timeframe Backtest, SMC & Options Engine")

# ==============================================================================
# 1. DISK + MEMORY CACHE & MULTI-SOURCE ANTI-RATE-LIMIT DATA ENGINE
# ==============================================================================
DATA_CACHE: Dict[str, pd.DataFrame] = {}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TIMEFRAME_HIERARCHY = ["1Y", "1M", "1W", "1d", "15m", "5m", "1m"]
NEXT_TIMEFRAME = {
    "1Y": "1M",
    "1M": "1W",
    "1W": "1d",
    "1d": "15m",
    "15m": "5m",
    "5m": "1m",
    "1m": "1m"
}

YF_LIMIT_DAYS = {
    "15m": 58,
    "5m": 58,
    "1m": 7
}


def _fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Free, un-throttled fallback for Daily historical OHLCV data via Stooq.
    Works immediately even when Yahoo Finance rate-limits your IP.
    """
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
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _fetch_nasdaq_daily(ticker: str) -> pd.DataFrame:
    """
    Secondary free fallback using Nasdaq's public chart API (up to 10 years of daily data).
    """
    clean_t = ticker.upper().strip()
    end_dt = datetime.now().strftime("%Y-%m-%d")
    start_dt = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    for asset_class in ["stocks", "etf"]:
        url = (
            f"https://api.nasdaq.com/api/quote/{clean_t}/historical"
            f"?assetclass={asset_class}&fromdate={start_dt}&limit=9999&todate={end_dt}"
        )
        r = requests.get(url, headers=headers, timeout=10)
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
                    return df[["open", "high", "low", "close", "volume"]].dropna()
    raise RuntimeError(f"Nasdaq API returned no data for {ticker}")


def _generate_intraday_from_daily(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """
    If Yahoo blocks intraday (15m/5m/1m), constructs realistic intraday bars
    anchored to the actual daily Open, High, Low, Close from Stooq/Nasdaq/Cache so you
    can still test drill-down, ORB, and VWAP strategies without interruption.
    """
    daily_csv = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
    if os.path.exists(daily_csv):
        df_1d = pd.read_csv(daily_csv, index_col=0, parse_dates=True).tail(days)
    else:
        try:
            df_1d = _fetch_stooq_daily(ticker).tail(days)
        except Exception:
            df_1d = _fetch_nasdaq_daily(ticker).tail(days)

    step_mins = {"15m": 15, "5m": 5, "1m": 1}.get(interval, 15)
    bars_per_day = int(390 / step_mins)
    all_rows = []
    np.random.seed(42)

    for dt, row in df_1d.iterrows():
        o, h, l, c, v = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])
        t_space = np.linspace(0, 1, bars_per_day)
        noise = np.cumsum(np.random.normal(0, 1, bars_per_day))
        noise = (noise - noise[0]) / (np.ptp(noise) + 1e-6)
        path = o + (c - o) * t_space + (h - l) * 0.35 * (noise - 0.5)

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
            u_weight = 1.8 if (b < 4 or b > bars_per_day - 4) else 0.7
            b_vol = max(100.0, (v / bars_per_day) * u_weight * np.random.uniform(0.7, 1.3))
            all_rows.append({
                "dt": bar_t,
                "open": b_open,
                "high": b_high,
                "low": b_low,
                "close": b_close,
                "volume": b_vol
            })

    return pd.DataFrame(all_rows).set_index("dt")


def _download_with_fallbacks(ticker: str, interval: str, period: str) -> tuple[pd.DataFrame, Optional[str]]:
    """
    1. Local Disk Cache -> 2. Yahoo Finance (native curl_cffi) ->
    3. Stooq / Nasdaq Real Daily Data -> 4. Anchored Intraday Reconstruction
    """
    csv_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")

    # 1. Load from disk cache if available
    if os.path.exists(csv_path):
        try:
            df_disk = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if not df_disk.empty:
                return df_disk, None
        except Exception:
            pass

    # 2. Try Yahoo Finance (no custom session passed so yfinance uses its native curl_cffi bypass)
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.to_csv(csv_path)
            return df, None
    except Exception:
        pass

    # 3. If Daily data requested and Yahoo is rate-limited, fetch real data from Stooq or Nasdaq
    if interval == "1d":
        try:
            df_stooq = _fetch_stooq_daily(ticker)
            df_stooq.to_csv(csv_path)
            return df_stooq, "Yahoo IP rate-limit bypassed: Loaded real historical daily data via Stooq."
        except Exception:
            try:
                df_nasdaq = _fetch_nasdaq_daily(ticker)
                df_nasdaq.to_csv(csv_path)
                return df_nasdaq, "Yahoo IP rate-limit bypassed: Loaded real historical daily data via Nasdaq API."
            except Exception as e_nas:
                return pd.DataFrame(), f"All daily data providers failed ({e_nas})."

    # 4. If Intraday (15m/5m/1m) and Yahoo is rate-limited, reconstruct from real Daily OHLCV
    try:
        days = YF_LIMIT_DAYS.get(interval, 7)
        df_syn = _generate_intraday_from_daily(ticker, interval, days)
        return df_syn, (
            f"Yahoo IP rate-limit active for '{interval}': Reconstructed {interval} bars "
            f"anchored to real Daily OHLCV so you can test drill-downs immediately."
        )
    except Exception as e:
        return pd.DataFrame(), f"Could not fetch or reconstruct {interval} data: {e}"


def fetch_and_construct_tf(ticker: str, tf: str) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Fetches Daily data and constructs 1W, 1M, 1Y from 1d data.
    Enforces lookback limits for 15m, 5m, 1m.
    """
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}_{tf}"

    if tf in ["1Y", "1M", "1W", "1d"]:
        base_key = f"{ticker}_1d"
        warn_msg = None
        if base_key not in DATA_CACHE:
            df_1d, warn_msg = _download_with_fallbacks(ticker, interval="1d", period="10y")
            if df_1d.empty:
                return pd.DataFrame(), warn_msg or f"No daily data found for {ticker}."
            DATA_CACHE[base_key] = df_1d

        df_base = DATA_CACHE[base_key].copy()

        if tf == "1d":
            return df_base, warn_msg

        rule_map = {"1W": "W-FRI", "1M": "ME", "1Y": "YE"}
        rule = rule_map[tf]
        try:
            df_res = df_base.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        except Exception:
            alt_rule = {"1W": "W", "1M": "M", "1Y": "Y"}[tf]
            df_res = df_base.resample(alt_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        return df_res, warn_msg

    else:
        max_days = YF_LIMIT_DAYS.get(tf, 7)
        warn_msg = None
        if cache_key not in DATA_CACHE:
            df_intra, warn_msg = _download_with_fallbacks(ticker, interval=tf, period=f"{max_days}d")
            if df_intra.empty:
                return pd.DataFrame(), warn_msg or f"No {tf} data found for {ticker}."
            DATA_CACHE[cache_key] = df_intra

        return DATA_CACHE[cache_key].copy(), warn_msg


# ==============================================================================
# 2. QUANTITATIVE INDICATORS & POINT-IN-TIME ENGINE
# ==============================================================================
def compute_all_indicators(
    df: pd.DataFrame,
    tf: str,
    atr_mult: float = 1.5,
    gk_mult: float = 2.0,
    w_trend: float = 0.4,
    w_liq: float = 0.4,
    w_rev: float = 0.2
) -> tuple[pd.DataFrame, List[str]]:
    """
    Computes all technical, volatility, liquidity, SMC/FVG, Fibonacci, ORB,
    and Regime Clarity indicators strictly point-in-time (zero lookahead).
    """
    df = df.copy()
    n = len(df)
    unavailable_indicators = []

    if n == 0:
        return df, ["All Indicators (Empty Data)"]

    # 1. Moving Averages
    if n < 20:
        unavailable_indicators.append("EMA 20 / Bollinger / GK Cone (Requires >= 20 bars)")
    if n < 50:
        unavailable_indicators.append("EMA 50 (Requires >= 50 bars)")

    df['ema20'] = df['close'].ewm(span=20, min_periods=min(5, n)).mean()
    df['ema50'] = df['close'].ewm(span=50, min_periods=min(10, n)).mean()

    # 2. True Range & ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14, min_periods=1).mean()

    # 3. Garman-Klass Realized Volatility & Next-Bar Prediction Interval
    log_hl = np.log((df['high'] / df['low'].replace(0, np.nan)).fillna(1.0))
    log_co = np.log((df['close'] / df['open'].replace(0, np.nan)).fillna(1.0))
    gk_var = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).clip(lower=1e-8)
    df['gk_vol_pct'] = np.sqrt(gk_var.ewm(span=20, min_periods=1).mean())
    df['gk_sigma_dollar'] = df['gk_vol_pct'] * df['close']

    # Next-bar projection cone anchored on current close + 3-bar drift
    drift = (df['close'] - df['close'].shift(3)).fillna(0) / 3.0
    df['gk_upper'] = df['close'] + drift + (gk_mult * df['gk_sigma_dollar'])
    df['gk_lower'] = df['close'] + drift - (gk_mult * df['gk_sigma_dollar'])

    df['atr_upper'] = df['close'] + (atr_mult * df['atr'])
    df['atr_lower'] = df['close'] - (atr_mult * df['atr'])

    # 4. Volume & RVOL
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['rvol'] = (df['volume'] / df['vol_sma20'].replace(0, 1)).fillna(1.0)

    # 5. VWAP & Standard Deviation Bands (Session anchored for intraday, 20-bar rolling for daily+)
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = typical_price * df['volume']

    if tf in ["15m", "5m", "1m"]:
        dates = df.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = df['volume'].groupby(dates).cumsum().replace(0, 1)
        df['vwap'] = cum_tp_vol / cum_vol
        std_vwap = typical_price.groupby(dates).transform(lambda x: x.expanding().std()).fillna(df['atr'] * 0.5)
    else:
        df['vwap'] = tp_vol.rolling(20, min_periods=1).sum() / df['volume'].rolling(20, min_periods=1).sum().replace(0, 1)
        std_vwap = df['close'].rolling(20, min_periods=1).std().fillna(df['atr'] * 0.5)

    df['vwap_u2'] = df['vwap'] + 2.0 * std_vwap
    df['vwap_l2'] = df['vwap'] - 2.0 * std_vwap

    # 6. RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 7. Point-in-Time Fibonacci Retracement (Rolling 50-bar Swing High / Low)
    swing_window = min(50, max(5, n))
    df['swing_high'] = df['high'].rolling(swing_window, min_periods=1).max()
    df['swing_low'] = df['low'].rolling(swing_window, min_periods=1).min()
    fib_range = (df['swing_high'] - df['swing_low']).replace(0, 1e-6)
    df['fib_382'] = df['swing_high'] - 0.382 * fib_range
    df['fib_500'] = df['swing_high'] - 0.500 * fib_range
    df['fib_618'] = df['swing_high'] - 0.618 * fib_range
    df['fib_786'] = df['swing_high'] - 0.786 * fib_range

    # 8. SMC: Fair Value Gaps (FVG) & Mitigation Retest
    body_prev = (df['close'].shift(1) - df['open'].shift(1)).abs()
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (body_prev > 0.8 * df['atr'])
    df['fvg_bull_top'] = np.where(df['fvg_bull'], df['low'], np.nan)
    df['fvg_bull_bot'] = np.where(df['fvg_bull'], df['high'].shift(2), np.nan)

    df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (body_prev > 0.8 * df['atr'])
    df['fvg_bear_top'] = np.where(df['fvg_bear'], df['low'].shift(2), np.nan)
    df['fvg_bear_bot'] = np.where(df['fvg_bear'], df['high'], np.nan)

    active_fvg_top = np.nan
    active_fvg_bot = np.nan
    fvg_mitigated_retest = np.zeros(n, dtype=bool)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    f_bull = df['fvg_bull'].values

    for i in range(2, n):
        if not np.isnan(active_fvg_top):
            if lows[i] <= active_fvg_top and closes[i] >= active_fvg_bot:
                fvg_mitigated_retest[i] = True
                active_fvg_top = np.nan
            elif closes[i] < active_fvg_bot:
                active_fvg_top = np.nan

        if f_bull[i]:
            active_fvg_top = lows[i]
            active_fvg_bot = highs[i - 2]

    df['smc_fvg_entry'] = fvg_mitigated_retest

    # 9. Opening Range Breakout (ORB) - First 30 mins of session (for intraday) or 5-bar box
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
        df['orb_high'] = df['high'].rolling(5, min_periods=1).max().shift(1)
        df['orb_low'] = df['low'].rolling(5, min_periods=1).min().shift(1)
        df['orb_mid'] = (df['orb_high'] + df['orb_low']) / 2.0

    # 10. Composite Score (alpha * Trend + beta * Liquidity + gamma * Reversion)
    slope_norm = np.clip(((df['ema20'] - df['ema50']) / df['atr'].replace(0, 1)), -2.0, 2.0) / 2.0
    bar_press = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)) * np.clip(df['rvol'] / 2.0, 0, 2.0)
    liq_norm = np.clip(bar_press, -1.0, 1.0)
    vwap_z = (df['close'] - df['vwap']) / std_vwap.replace(0, 1)
    rev_norm = np.clip(-vwap_z / 2.5, -1.0, 1.0)

    total_w = abs(w_trend) + abs(w_liq) + abs(w_rev)
    if total_w == 0:
        total_w = 1.0
    df['composite_score'] = (w_trend * slope_norm + w_liq * liq_norm + w_rev * rev_norm) / total_w

    # 11. Unclear / Volatile Chop Detection (Gray Shaded Regime)
    ema_diff_ratio = (df['ema20'] - df['ema50']).abs() / df['atr'].replace(0, 1)
    df['is_unclear_volatile'] = (ema_diff_ratio < 0.32) | (
        (df['atr'] > 1.35 * df['atr'].rolling(20, min_periods=1).mean()) & (df['composite_score'].abs() < 0.20)
    )

    return df, unavailable_indicators


# ==============================================================================
# 3. 4-STATE LIFECYCLE BACKTEST ENGINE ($100k Baseline + Portfolio Sizing)
# ==============================================================================
STRATEGY_METADATA = {
    "trend": {
        "name": "1. Core Trend (EMA 20/50 + Volume)",
        "indicators": ["ema20", "ema50", "atr_bands"],
        "bucket": "Core 70%"
    },
    "watch_dip": {
        "name": "2. Watch Signal -> Wait for Dip -> Buy",
        "indicators": ["ema20", "ema50", "vwap", "gk_bands"],
        "bucket": "Core 70%"
    },
    "smc_fvg": {
        "name": "3. SMC & Fair Value Gaps (FVG Retest)",
        "indicators": ["fvg_zones", "ema50", "atr_bands"],
        "bucket": "Satellite 30%"
    },
    "fib_golden": {
        "name": "4. Fibonacci Golden Pocket (0.618 Pullback)",
        "indicators": ["fib_levels", "ema50"],
        "bucket": "Core 70%"
    },
    "orb": {
        "name": "5. Opening Range Breakout (30m ORB + VWAP)",
        "indicators": ["orb_levels", "vwap"],
        "bucket": "Satellite 30%"
    },
    "do_t": {
        "name": "6. Intraday 'Do T' (VWAP +-2 Sigma Reversion)",
        "indicators": ["vwap", "vwap_bands", "gk_bands"],
        "bucket": "Satellite 30%"
    },
    "composite": {
        "name": "7. Composite Score (Alpha*A + Beta*B + Gamma*C)",
        "indicators": ["ema20", "vwap", "gk_bands"],
        "bucket": "Core 70%"
    },
    "plain_tpsl": {
        "name": "8. Plain TP/SL & Manual Click-to-Buy Sandbox",
        "indicators": ["atr_bands", "gk_bands"],
        "bucket": "Sandbox"
    }
}


def run_strategy_backtest(
    df: pd.DataFrame,
    strategy_key: str,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    vol_model: str = "atr",
    initial_capital: float = 100000.0,
    core_pct: float = 0.70,
    intraday_pct: float = 0.30,
    risk_per_trade_pct: float = 0.01,
    manual_entry_ts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the 4-State Lifecycle Machine:
    0 = IDLE, 1 = WATCHING, 2 = IN_POSITION, 3 = STOPPED_OUT (Re-entry armed)
    """
    n = len(df)
    if n == 0:
        return {"trades": [], "markers": [], "boxes": [], "stats": {}, "equity": []}

    timestamps = [int(ts.timestamp()) for ts in df.index]
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values
    gks = df['gk_sigma_dollar'].values

    ema20 = df['ema20'].values
    ema50 = df['ema50'].values
    rvol = df['rvol'].values
    vwap = df['vwap'].values
    vwap_l2 = df['vwap_l2'].values
    fib_618 = df['fib_618'].values
    fib_786 = df['fib_786'].values
    orb_high = df['orb_high'].values
    orb_mid = df['orb_mid'].values
    smc_fvg = df['smc_fvg_entry'].values
    comp_score = df['composite_score'].values
    rsi = df['rsi'].values

    is_intraday_strat = strategy_key in ["orb", "do_t", "smc_fvg"]
    pool_fraction = intraday_pct if is_intraday_strat else core_pct

    equity = initial_capital
    peak_equity = initial_capital
    max_dd_pct = 0.0

    state = 0  # 0: IDLE, 1: WATCHING, 2: IN_POSITION, 3: STOPPED_OUT
    watch_peak_price = 0.0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    alt_sl_price = 0.0
    alt_tp_price = 0.0
    entry_idx = 0
    position_shares = 0.0

    trades = []
    markers = []
    boxes = []
    equity_curve = []

    for i in range(1, n):
        ts = timestamps[i]
        c_high, c_low, c_close, c_open = highs[i], lows[i], closes[i], opens[i]
        unit_vol = gks[i] if vol_model == "gk" else atrs[i]
        if np.isnan(unit_vol) or unit_vol <= 0:
            unit_vol = c_close * 0.015

        trend_on = False
        trigger_buy = False

        if strategy_key == "trend":
            trend_on = c_close > ema50[i] and ema20[i] > ema50[i]
            trigger_buy = trend_on and (c_low <= ema20[i] * 1.008) and (c_close > c_open) and (rvol[i] >= 0.9)

        elif strategy_key == "watch_dip":
            trend_on = ema20[i] > ema50[i]
            if state in [1, 3]:
                watch_peak_price = max(watch_peak_price, c_high)
                dip_distance = watch_peak_price - c_low
                trigger_buy = (dip_distance >= 0.8 * unit_vol) and (c_close > c_open)

        elif strategy_key == "smc_fvg":
            trend_on = c_close > ema50[i] * 0.98
            trigger_buy = bool(smc_fvg[i])

        elif strategy_key == "fib_golden":
            trend_on = c_close > fib_786[i] and ema20[i] >= ema50[i] * 0.99
            trigger_buy = trend_on and (c_low <= fib_618[i] * 1.005) and (c_close >= fib_618[i]) and (rsi[i] < 60)

        elif strategy_key == "orb":
            if not np.isnan(orb_high[i]):
                trend_on = c_close > orb_mid[i] and c_close > vwap[i]
                trigger_buy = (closes[i - 1] <= orb_high[i - 1]) and (c_close > orb_high[i]) and (rvol[i] >= 1.2)

        elif strategy_key == "do_t":
            trend_on = c_close > ema50[i] * 0.95
            trigger_buy = (c_low <= vwap_l2[i]) and (c_close > c_open)

        elif strategy_key == "composite":
            trend_on = comp_score[i] > 0.15
            trigger_buy = comp_score[i] >= 0.48

        elif strategy_key == "plain_tpsl":
            if manual_entry_ts is not None:
                trend_on = True
                trigger_buy = (ts >= manual_entry_ts) and (timestamps[i - 1] < manual_entry_ts or i == 1)
            else:
                trend_on = c_close > ema20[i]
                trigger_buy = (c_close > highs[i - 1]) and (c_close > c_open)

        # ---------------- STATE MACHINE EXECUTION ----------------
        if state == 0:
            if strategy_key == "plain_tpsl" and manual_entry_ts is not None and trigger_buy:
                state = 2
                entry_price = c_close
                entry_idx = i
                sl_price = entry_price - sl_mult * atrs[i]
                tp_price = entry_price + tp_mult * atrs[i]
                alt_sl_price = entry_price - sl_mult * gks[i]
                alt_tp_price = entry_price + tp_mult * gks[i]
                risk_dollars = equity * risk_per_trade_pct
                max_alloc_dollars = equity * pool_fraction
                position_shares = min(risk_dollars / max(entry_price - sl_price, 0.01), max_alloc_dollars / entry_price)
                markers.append({
                    "time": ts, "position": "belowBar", "color": "#00E676",
                    "shape": "arrowUp", "text": f"MANUAL BUY @ {entry_price:.2f}"
                })
            elif trend_on:
                state = 1
                watch_peak_price = c_high
                markers.append({
                    "time": ts, "position": "belowBar", "color": "#90A4AE",
                    "shape": "circle", "text": "WATCH"
                })

        elif state == 1:
            if not trend_on and manual_entry_ts is None:
                state = 0
            elif trigger_buy:
                state = 2
                entry_price = c_close
                entry_idx = i
                sl_dist = sl_mult * unit_vol
                tp_dist = tp_mult * unit_vol

                if strategy_key == "do_t":
                    sl_price = entry_price - (0.85 * unit_vol)
                    tp_price = max(vwap[i], entry_price + 0.8 * unit_vol)
                elif strategy_key == "orb" and not np.isnan(orb_mid[i]):
                    sl_price = min(orb_mid[i], entry_price - 0.5 * unit_vol)
                    tp_price = entry_price + tp_mult * max(entry_price - sl_price, 0.5 * unit_vol)
                else:
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist

                alt_sl_price = entry_price - sl_mult * gks[i]
                alt_tp_price = entry_price + tp_mult * gks[i]

                risk_dollars = equity * risk_per_trade_pct
                max_alloc_dollars = equity * pool_fraction
                stop_gap = max(entry_price - sl_price, entry_price * 0.005)
                position_shares = min(risk_dollars / stop_gap, max_alloc_dollars / entry_price)

                markers.append({
                    "time": ts, "position": "belowBar", "color": "#00E676",
                    "shape": "arrowUp", "text": f"BUY @ {entry_price:.2f}"
                })

        elif state == 2:
            hit_tp = c_high >= tp_price
            hit_sl = c_low <= sl_price

            if hit_tp or hit_sl:
                exit_price = tp_price if hit_tp else sl_price
                pnl = (exit_price - entry_price) * position_shares
                ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
                equity += pnl

                trades.append({
                    "entry_time": timestamps[entry_idx],
                    "exit_time": ts,
                    "type": "TP" if hit_tp else "SL",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": round(position_shares, 2),
                    "pnl": round(pnl, 2),
                    "ret_pct": round(ret_pct, 2)
                })

                boxes.append({
                    "t1": timestamps[entry_idx],
                    "t2": ts,
                    "entry": float(entry_price),
                    "tp": float(tp_price),
                    "sl": float(sl_price),
                    "alt_tp": float(alt_tp_price),
                    "alt_sl": float(alt_sl_price),
                    "outcome": "TP" if hit_tp else "SL"
                })

                if hit_tp:
                    markers.append({
                        "time": ts, "position": "aboveBar", "color": "#FFD600",
                        "shape": "arrowDown", "text": f"TP (+${pnl:,.0f})"
                    })
                    state = 1 if trend_on else 0
                    watch_peak_price = c_high
                else:
                    markers.append({
                        "time": ts, "position": "aboveBar", "color": "#FF1744",
                        "shape": "arrowDown", "text": f"SL (-${abs(pnl):,.0f})"
                    })
                    state = 3 if trend_on else 0
                    watch_peak_price = c_high

        elif state == 3:
            if not trend_on:
                state = 0
            elif trigger_buy:
                state = 2
                entry_price = c_close
                entry_idx = i
                sl_price = entry_price - (sl_mult * unit_vol)
                tp_price = entry_price + (tp_mult * unit_vol)
                alt_sl_price = entry_price - sl_mult * gks[i]
                alt_tp_price = entry_price + tp_mult * gks[i]

                risk_dollars = equity * risk_per_trade_pct
                max_alloc_dollars = equity * pool_fraction
                stop_gap = max(entry_price - sl_price, entry_price * 0.005)
                position_shares = min(risk_dollars / stop_gap, max_alloc_dollars / entry_price)

                markers.append({
                    "time": ts, "position": "belowBar", "color": "#00B0FF",
                    "shape": "arrowUp", "text": f"RE-BUY @ {entry_price:.2f}"
                })

        peak_equity = max(peak_equity, equity)
        dd = ((peak_equity - equity) / peak_equity) * 100.0
        max_dd_pct = max(max_dd_pct, dd)
        equity_curve.append({"time": ts, "value": round(equity, 2)})

    if state == 2 and n > 0:
        boxes.append({
            "t1": timestamps[entry_idx],
            "t2": timestamps[-1],
            "entry": float(entry_price),
            "tp": float(tp_price),
            "sl": float(sl_price),
            "alt_tp": float(alt_tp_price),
            "alt_sl": float(alt_sl_price),
            "outcome": "ACTIVE"
        })

    total_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    net_return_pct = ((equity - initial_capital) / initial_capital) * 100.0
    buy_hold_ret = ((closes[-1] - closes[0]) / closes[0]) * 100.0 if n > 1 else 0.0

    stats = {
        "strategy_key": strategy_key,
        "strategy_name": STRATEGY_METADATA[strategy_key]["name"],
        "bucket": STRATEGY_METADATA[strategy_key]["bucket"],
        "initial_capital": initial_capital,
        "ending_equity": round(equity, 2),
        "net_pnl": round(equity - initial_capital, 2),
        "net_return_pct": round(net_return_pct, 2),
        "buy_hold_pct": round(buy_hold_ret, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(min(profit_factor, 99.9), 2),
        "max_dd_pct": round(max_dd_pct, 2)
    }

    return {
        "trades": trades,
        "markers": markers,
        "boxes": boxes,
        "stats": stats,
        "equity": equity_curve
    }


# ==============================================================================
# 4. BLACK-SCHOLES GREEKS & IMPLIED EXPECTANCY ENGINE
# ==============================================================================
def black_scholes_analysis(
    S: float, K: float, days_to_expiry: int, iv_pct: float, r_pct: float, gk_daily_vol_pct: float, opt_type: str = "call"
) -> Dict[str, Any]:
    T = max(days_to_expiry / 365.0, 1 / 3650.0)
    sigma = max(iv_pct / 100.0, 0.01)
    r = r_pct / 100.0

    hv_annual = gk_daily_vol_pct * math.sqrt(252)

    d1 = (math.log(S / max(K, 0.01)) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
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

    return {
        "theo_price": round(price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 4),
        "theta_daily": round(theta, 3),
        "vega_per_1pct": round(vega, 3),
        "prob_itm_pct": round(prob_itm, 1),
        "breakeven": round(breakeven, 2),
        "hv_annual_pct": round(hv_annual * 100.0, 1),
        "iv_hv_ratio": round(iv_hv_ratio, 2),
        "expected_move_iv": round(expected_move_iv_dollar, 2),
        "expected_move_hv": round(expected_move_hv_dollar, 2),
        "verdict": verdict
    }


# ==============================================================================
# 5. FASTAPI ENDPOINTS
# ==============================================================================
@app.get("/api/analyze")
def api_analyze(
    ticker: str = Query("SNDK"),
    tf: str = Query("1d"),
    start_ts: Optional[int] = Query(None),
    cutoff_ts: Optional[int] = Query(None),
    strategy: str = Query("trend"),
    sl_mult: float = Query(1.5),
    tp_mult: float = Query(3.0),
    vol_model: str = Query("atr"),
    w_trend: float = Query(0.4),
    w_liq: float = Query(0.4),
    w_rev: float = Query(0.2),
    initial_capital: float = Query(100000.0),
    core_pct: float = Query(70.0),
    intraday_pct: float = Query(30.0),
    risk_pct: float = Query(1.0),
    manual_entry_ts: Optional[int] = Query(None)
):
    df_raw, warn_msg = fetch_and_construct_tf(ticker, tf)
    if df_raw.empty:
        return JSONResponse({"error": warn_msg or "No data available."}, status_code=400)

    first_avail_ts = int(df_raw.index[0].timestamp())
    if start_ts is not None and start_ts < first_avail_ts:
        first_dt_str = df_raw.index[0].strftime("%Y-%m-%d %H:%M")
        extra_warn = (
            f"Intraday limit: '{tf}' data is unavailable for the clicked date. "
            f"Showing earliest available '{tf}' data starting from {first_dt_str}."
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
            warmup_start_idx = max(0, first_idx - 100)
            df_calc = df_raw.iloc[warmup_start_idx:].copy()
        else:
            df_calc = df_raw.tail(200).copy()
    else:
        df_calc = df_raw.copy()

    df_ind, unavailable_list = compute_all_indicators(
        df_calc, tf, atr_mult=sl_mult, gk_mult=tp_mult, w_trend=w_trend, w_liq=w_liq, w_rev=w_rev
    )

    if start_ts is not None:
        start_dt = datetime.fromtimestamp(start_ts)
        df_view = df_ind[df_ind.index >= start_dt].copy()
        if df_view.empty:
            df_view = df_ind.copy()
    else:
        df_view = df_ind.tail(500).copy()

    bt_selected = run_strategy_backtest(
        df_view,
        strategy_key=strategy,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        vol_model=vol_model,
        initial_capital=initial_capital,
        core_pct=core_pct / 100.0,
        intraday_pct=intraday_pct / 100.0,
        risk_per_trade_pct=risk_pct / 100.0,
        manual_entry_ts=manual_entry_ts
    )

    all_summary = []
    for s_key in STRATEGY_METADATA.keys():
        res = run_strategy_backtest(
            df_view,
            strategy_key=s_key,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            vol_model=vol_model,
            initial_capital=initial_capital,
            core_pct=core_pct / 100.0,
            intraday_pct=intraday_pct / 100.0,
            risk_per_trade_pct=risk_pct / 100.0,
            manual_entry_ts=manual_entry_ts if s_key == "plain_tpsl" else None
        )
        all_summary.append(res["stats"])

    candles = []
    volumes = []
    series_data = {
        "ema20": [], "ema50": [], "vwap": [], "vwap_u2": [], "vwap_l2": [],
        "atr_upper": [], "atr_lower": [], "gk_upper": [], "gk_lower": [],
        "fib_382": [], "fib_500": [], "fib_618": [], "fib_786": [],
        "orb_high": [], "orb_low": []
    }
    fvg_boxes = []
    unclear_spans = []

    in_unclear = False
    unclear_start = None
    idx_timestamps = [int(t.timestamp()) for t in df_view.index]

    for i, (dt, row) in enumerate(df_view.iterrows()):
        ts = idx_timestamps[i]
        candles.append({
            "time": ts,
            "open": round(float(row['open']), 4),
            "high": round(float(row['high']), 4),
            "low": round(float(row['low']), 4),
            "close": round(float(row['close']), 4)
        })
        volumes.append({
            "time": ts,
            "value": float(row['volume']),
            "color": "rgba(0, 230, 118, 0.35)" if row['close'] >= row['open'] else "rgba(255, 23, 68, 0.35)"
        })

        for col in series_data.keys():
            val = row.get(col, np.nan)
            if not np.isnan(val):
                series_data[col].append({"time": ts, "value": round(float(val), 4)})

        if row['fvg_bull'] and i >= 2:
            t_end = idx_timestamps[min(i + 8, len(idx_timestamps) - 1)]
            fvg_boxes.append({
                "t1": idx_timestamps[i - 2], "t2": t_end,
                "top": float(row['fvg_bull_top']), "bot": float(row['fvg_bull_bot']),
                "kind": "BULL_FVG"
            })
        if row['fvg_bear'] and i >= 2:
            t_end = idx_timestamps[min(i + 8, len(idx_timestamps) - 1)]
            fvg_boxes.append({
                "t1": idx_timestamps[i - 2], "t2": t_end,
                "top": float(row['fvg_bear_top']), "bot": float(row['fvg_bear_bot']),
                "kind": "BEAR_FVG"
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
    last_close = float(last_row['close'])
    last_atr = float(last_row['atr'])
    last_gk_pct = float(last_row['gk_vol_pct'])

    return {
        "ticker": ticker.upper(),
        "tf": tf,
        "next_tf": NEXT_TIMEFRAME.get(tf, "1m"),
        "warning": warn_msg,
        "unavailable_indicators": unavailable_list,
        "auto_indicators": STRATEGY_METADATA[strategy]["indicators"],
        "candles": candles,
        "volumes": volumes,
        "series": series_data,
        "fvg_boxes": fvg_boxes[-40:],
        "unclear_spans": unclear_spans,
        "backtest": bt_selected,
        "all_strategies_summary": all_summary,
        "latest_metrics": {
            "close": round(last_close, 2),
            "atr": round(last_atr, 2),
            "gk_daily_vol_pct": round(last_gk_pct * 100.0, 2),
            "composite_score": round(float(last_row['composite_score']), 2)
        }
    }


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
# 6. FRONTEND SINGLE-PAGE APPLICATION (TradingView Lightweight Charts + Canvas)
# ==============================================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quant Strategy, Drill-Down & Options Terminal</title>
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
  
  /* Left Sidebar */
  .sidebar { width: 340px; background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 12px; }
  .card { background: #1a1f2e; border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .card-title { font-weight: 700; font-size: 12px; text-transform: uppercase; color: #90a4ae; margin-bottom: 8px; letter-spacing: 0.5px; display: flex; justify-content: space-between; }
  
  label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 3px; margin-top: 6px; }
  input[type="text"], input[type="number"], select {
    width: 100%; background: #0b0e14; border: 1px solid var(--border); color: var(--text);
    padding: 6px 8px; border-radius: 4px; font-size: 12px;
  }
  .row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  
  .chk-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 4px; }
  .chk-item { display: flex; align-items: center; gap: 6px; font-size: 11px; cursor: pointer; }
  
  button.btn {
    background: var(--accent); color: #fff; border: none; border-radius: 4px;
    padding: 7px 10px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 8px;
  }
  button.btn:hover { opacity: 0.9; }
  button.btn-sec { background: #2a2e39; color: var(--text); }
  
  /* Main Content Area */
  .main { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
  
  /* Top Breadcrumb & Ribbon */
  .topbar {
    background: var(--panel); border-bottom: 1px solid var(--border); padding: 8px 14px;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px;
  }
  .breadcrumbs { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .crumb {
    background: #1e2433; border: 1px solid #363c4e; color: var(--text);
    padding: 4px 9px; border-radius: 4px; font-size: 11px; cursor: pointer;
  }
  .crumb.active { background: var(--accent); border-color: var(--accent); font-weight: 700; }
  .tf-selector { display: flex; gap: 4px; }
  .tf-btn {
    background: #1a1f2e; border: 1px solid var(--border); color: var(--muted);
    padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 600;
  }
  .tf-btn.active { background: #00b0ff; color: #000; border-color: #00b0ff; }
  
  .alert-banner {
    background: rgba(255, 152, 0, 0.15); border-bottom: 1px solid #ff9800;
    color: #ffb74d; padding: 6px 14px; font-size: 12px; display: none;
  }

  /* Chart Stack */
  .chart-wrapper { position: relative; width: 100%; height: 520px; background: #0b0e14; border-bottom: 1px solid var(--border); }
  #tv-chart { width: 100%; height: 100%; }
  #overlay-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }
  .chart-legend {
    position: absolute; top: 10px; left: 14px; z-index: 20; background: rgba(19, 23, 34, 0.82);
    padding: 6px 10px; border-radius: 4px; border: 1px solid var(--border); pointer-events: none; font-size: 11px;
  }

  /* Bottom Tables & Options Panel */
  .bottom-grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 12px; padding: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; background: #161b26; }
  tr.highlight-row { background: rgba(41, 98, 255, 0.18); font-weight: 600; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
</style>
</head>
<body>

<!-- ==================== LEFT SIDEBAR ==================== -->
<div class="sidebar">
  <div class="card">
    <div class="card-title">1. Asset & Click Action</div>
    <div class="row-2">
      <div>
        <label>Ticker (e.g. SNDK, NVDA, ASTS)</label>
        <input type="text" id="ticker" value="SNDK">
      </div>
      <div>
        <label>Candle Click Mode</label>
        <select id="clickMode">
          <option value="drill">Drill-Down Timeframe</option>
          <option value="manual_buy">Manual Buy (TP/SL)</option>
          <option value="cutoff">Set Point-in-Time Cutoff</option>
        </select>
      </div>
    </div>
    <div class="row-2" style="margin-top:4px;">
      <button class="btn" onclick="resetToRoot()">Load Ticker</button>
      <button class="btn btn-sec" onclick="clearCutoffAndManual()">Reset Cutoff/Buy</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">2. Strategy & Auto-Indicators</div>
    <label>Active Strategy</label>
    <select id="strategy" onchange="onStrategyChange()">
      <option value="trend">1. Core Trend (EMA 20/50 + Volume)</option>
      <option value="watch_dip">2. Watch Signal -> Wait for Dip -> Buy</option>
      <option value="smc_fvg">3. SMC & Fair Value Gaps (FVG Retest)</option>
      <option value="fib_golden">4. Fibonacci Golden Pocket (0.618)</option>
      <option value="orb">5. Opening Range Breakout (30m ORB)</option>
      <option value="do_t">6. Intraday 'Do T' (VWAP +-2 Sigma)</option>
      <option value="composite">7. Composite Score (a*A + b*B + c*C)</option>
      <option value="plain_tpsl">8. Plain TP/SL & Manual Click Sandbox</option>
    </select>

    <div class="row-3" style="margin-top:6px;">
      <div>
        <label>SL Mult</label>
        <input type="number" id="sl_mult" value="1.5" step="0.25">
      </div>
      <div>
        <label>TP Mult</label>
        <input type="number" id="tp_mult" value="3.0" step="0.25">
      </div>
      <div>
        <label>Band Model</label>
        <select id="vol_model">
          <option value="atr">ATR (14)</option>
          <option value="gk">Garman-Klass</option>
        </select>
      </div>
    </div>

    <label style="margin-top:8px;">Composite Weights (Alpha / Beta / Gamma)</label>
    <div class="row-3">
      <div><label>a (Trend)</label><input type="number" id="w_trend" value="0.4" step="0.1"></div>
      <div><label>b (Liquid)</label><input type="number" id="w_liq" value="0.4" step="0.1"></div>
      <div><label>g (VWAP)</label><input type="number" id="w_rev" value="0.2" step="0.1"></div>
    </div>
    <button class="btn" onclick="loadChartData()">Apply Parameters</button>
  </div>

  <div class="card">
    <div class="card-title">3. Selectable Overlays & Zones</div>
    <div class="chk-grid">
      <label class="chk-item"><input type="checkbox" id="chk_ema20" checked onchange="renderOverlays()"> EMA 20 & 50</label>
      <label class="chk-item"><input type="checkbox" id="chk_vwap" onchange="renderOverlays()"> Session VWAP</label>
      <label class="chk-item"><input type="checkbox" id="chk_vwap_bands" onchange="renderOverlays()"> VWAP +-2s Bands</label>
      <label class="chk-item"><input type="checkbox" id="chk_atr_bands" checked onchange="renderOverlays()"> ATR Interval</label>
      <label class="chk-item"><input type="checkbox" id="chk_gk_bands" checked onchange="renderOverlays()"> Garman-Klass Cone</label>
      <label class="chk-item"><input type="checkbox" id="chk_fib_levels" onchange="renderOverlays()"> Fibonacci Levels</label>
      <label class="chk-item"><input type="checkbox" id="chk_orb_levels" onchange="renderOverlays()"> ORB High/Low</label>
      <label class="chk-item"><input type="checkbox" id="chk_fvg_zones" checked onchange="renderOverlays()"> SMC FVG Boxes</label>
      <label class="chk-item"><input type="checkbox" id="chk_tpsl_boxes" checked onchange="renderOverlays()"> Shaded TP/SL Boxes</label>
      <label class="chk-item"><input type="checkbox" id="chk_unclear" checked onchange="renderOverlays()"> Gray Volatile Area</label>
    </div>
  </div>

  <div class="card">
    <div class="card-title">4. Portfolio & Position Sizing ($100k)</div>
    <div class="row-2">
      <div>
        <label>Starting Capital ($)</label>
        <input type="number" id="initial_capital" value="100000" step="10000">
      </div>
      <div>
        <label>Risk / Trade (%)</label>
        <input type="number" id="risk_pct" value="1.0" step="0.25">
      </div>
    </div>
    <div class="row-2">
      <div>
        <label>Core Trend Pool (%)</label>
        <input type="number" id="core_pct" value="70" step="5">
      </div>
      <div>
        <label>Intraday / T Pool (%)</label>
        <input type="number" id="intraday_pct" value="30" step="5">
      </div>
    </div>
  </div>
</div>

<!-- ==================== MAIN VIEWPORT ==================== -->
<div class="main">
  <!-- Top Navigation & Breadcrumbs -->
  <div class="topbar">
    <div class="breadcrumbs" id="breadcrumb-bar"></div>
    <div class="tf-selector" id="tf-buttons">
      <button class="tf-btn" onclick="switchTF('1Y')">1Y</button>
      <button class="tf-btn" onclick="switchTF('1M')">1M</button>
      <button class="tf-btn" onclick="switchTF('1W')">1W</button>
      <button class="tf-btn active" onclick="switchTF('1d')">1D</button>
      <button class="tf-btn" onclick="switchTF('15m')">15m</button>
      <button class="tf-btn" onclick="switchTF('5m')">5m</button>
      <button class="tf-btn" onclick="switchTF('1m')">1m</button>
    </div>
  </div>

  <div class="alert-banner" id="alert-banner"></div>

  <!-- Interactive Chart + Shaded Canvas -->
  <div class="chart-wrapper">
    <div class="chart-legend" id="chart-legend">Loading...</div>
    <div id="tv-chart"></div>
    <canvas id="overlay-canvas"></canvas>
  </div>

  <!-- Bottom Summary Tables & Options Calculator -->
  <div class="bottom-grid">
    <div class="card">
      <div class="card-title">
        <span>All Strategies Backtest Comparison ($100k Baseline on Current View)</span>
        <span id="bh-badge" style="color:var(--gold)"></span>
      </div>
      <div style="overflow-x:auto;">
        <table id="summary-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Pool</th>
              <th>Ending Equity</th>
              <th>Net Return</th>
              <th>Trades</th>
              <th>Win Rate</th>
              <th>Profit Factor</th>
              <th>Max DD</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Options Greeks & Implied Expectancy Engine</div>
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
      <div id="opt-results" style="margin-top:8px; font-size:11.5px; line-height:1.6;"></div>
    </div>
  </div>
</div>

<script>
let chart, candleSeries, volumeSeries;
let lineSeriesMap = {};
let currentData = null;

// Navigation State
let currentTF = "1d";
let currentStartTs = null;
let currentCutoffTs = null;
let manualEntryTs = null;
let breadcrumbs = [{ label: "All History (1D)", tf: "1d", startTs: null }];

function initChart() {
  const container = document.getElementById('tv-chart');
  chart = LightweightCharts.createChart(container, {
    layout: { background: { type: 'solid', color: '#0b0e14' }, textColor: '#d1d4dc' },
    grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2e39' },
    timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false }
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#00e676', downColor: '#ff1744',
    borderUpColor: '#00e676', borderDownColor: '#ff1744',
    wickUpColor: '#00e676', wickDownColor: '#ff1744'
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: ''
  });
  volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

  const lineConfigs = {
    ema20: { color: '#2962ff', lineWidth: 2, title: 'EMA 20' },
    ema50: { color: '#ff9800', lineWidth: 2, title: 'EMA 50' },
    vwap: { color: '#e040fb', lineWidth: 2, title: 'VWAP' },
    vwap_u2: { color: 'rgba(224,64,251,0.5)', lineWidth: 1, lineStyle: 2, title: 'VWAP +2s' },
    vwap_l2: { color: 'rgba(224,64,251,0.5)', lineWidth: 1, lineStyle: 2, title: 'VWAP -2s' },
    atr_upper: { color: 'rgba(0,229,255,0.6)', lineWidth: 1, lineStyle: 2, title: 'ATR Upper' },
    atr_lower: { color: 'rgba(0,229,255,0.6)', lineWidth: 1, lineStyle: 2, title: 'ATR Lower' },
    gk_upper: { color: 'rgba(255,214,0,0.7)', lineWidth: 1, lineStyle: 1, title: 'GK Cone Upper' },
    gk_lower: { color: 'rgba(255,214,0,0.7)', lineWidth: 1, lineStyle: 1, title: 'GK Cone Lower' },
    fib_382: { color: 'rgba(144,164,174,0.5)', lineWidth: 1, title: 'Fib 0.382' },
    fib_500: { color: 'rgba(144,164,174,0.5)', lineWidth: 1, title: 'Fib 0.500' },
    fib_618: { color: '#ffd600', lineWidth: 2, title: 'Fib 0.618 Golden' },
    fib_786: { color: 'rgba(255,23,68,0.5)', lineWidth: 1, title: 'Fib 0.786' },
    orb_high: { color: '#00e676', lineWidth: 1, lineStyle: 2, title: 'ORB High' },
    orb_low: { color: '#ff1744', lineWidth: 1, lineStyle: 2, title: 'ORB Low' }
  };

  for (const [k, cfg] of Object.entries(lineConfigs)) {
    lineSeriesMap[k] = chart.addLineSeries({ ...cfg, lastValueVisible: false, priceLineVisible: false });
  }

  chart.timeScale().subscribeVisibleTimeRangeChange(() => requestAnimationFrame(drawCanvasZones));
  window.addEventListener('resize', () => {
    const wrap = document.querySelector('.chart-wrapper');
    chart.applyOptions({ width: wrap.clientWidth, height: wrap.clientHeight });
    requestAnimationFrame(drawCanvasZones);
  });

  // Handle Candlestick Click -> Drill-Down or Manual Buy or Point-in-Time Cutoff
  chart.subscribeClick((param) => {
    if (!param || !param.time || !currentData) return;
    const clickedTs = param.time;
    const mode = document.getElementById('clickMode').value;
    const dtStr = new Date(clickedTs * 1000).toISOString().slice(0, 16).replace('T', ' ');

    if (mode === 'drill') {
      const nextTf = currentData.next_tf;
      if (nextTf === currentTF && currentTF === '1m') return;
      currentTF = nextTf;
      currentStartTs = clickedTs;
      breadcrumbs.push({ label: `${dtStr} (${currentTF})`, tf: currentTF, startTs: currentStartTs });
      loadChartData();
    } else if (mode === 'manual_buy') {
      document.getElementById('strategy').value = 'plain_tpsl';
      manualEntryTs = clickedTs;
      loadChartData();
    } else if (mode === 'cutoff') {
      currentCutoffTs = clickedTs;
      loadChartData();
    }
  });
}

function renderBreadcrumbs() {
  const bar = document.getElementById('breadcrumb-bar');
  bar.innerHTML = '';
  breadcrumbs.forEach((b, idx) => {
    const el = document.createElement('span');
    el.className = 'crumb' + (idx === breadcrumbs.length - 1 ? ' active' : '');
    el.textContent = b.label;
    el.onclick = () => {
      breadcrumbs = breadcrumbs.slice(0, idx + 1);
      currentTF = b.tf;
      currentStartTs = b.startTs;
      loadChartData();
    };
    bar.appendChild(el);
  });

  document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.toLowerCase() === currentTF.toLowerCase());
  });
}

function switchTF(tf) {
  currentTF = tf;
  breadcrumbs.push({ label: `View (${tf})`, tf: tf, startTs: currentStartTs });
  loadChartData();
}

function resetToRoot() {
  currentTF = '1Y';
  currentStartTs = null;
  currentCutoffTs = null;
  manualEntryTs = null;
  breadcrumbs = [{ label: `All Years (1Y)`, tf: '1Y', startTs: null }];
  loadChartData();
}

function clearCutoffAndManual() {
  currentCutoffTs = null;
  manualEntryTs = null;
  loadChartData();
}

function onStrategyChange() {
  loadChartData();
}

async function loadChartData() {
  renderBreadcrumbs();
  const params = new URLSearchParams({
    ticker: document.getElementById('ticker').value,
    tf: currentTF,
    strategy: document.getElementById('strategy').value,
    sl_mult: document.getElementById('sl_mult').value,
    tp_mult: document.getElementById('tp_mult').value,
    vol_model: document.getElementById('vol_model').value,
    w_trend: document.getElementById('w_trend').value,
    w_liq: document.getElementById('w_liq').value,
    w_rev: document.getElementById('w_rev').value,
    initial_capital: document.getElementById('initial_capital').value,
    core_pct: document.getElementById('core_pct').value,
    intraday_pct: document.getElementById('intraday_pct').value,
    risk_pct: document.getElementById('risk_pct').value
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
    msgs.push("Hidden due to insufficient warm-up bars: " + data.unavailable_indicators.join(", "));
  }
  if (msgs.length > 0) {
    banner.style.display = 'block';
    banner.textContent = msgs.join(" | ");
  } else {
    banner.style.display = 'none';
  }

  const allIndIds = ['ema20', 'vwap', 'vwap_bands', 'atr_bands', 'gk_bands', 'fib_levels', 'orb_levels', 'fvg_zones'];
  allIndIds.forEach(id => {
    const el = document.getElementById('chk_' + id);
    if (el) el.checked = data.auto_indicators.includes(id);
  });

  candleSeries.setData(data.candles);
  volumeSeries.setData(data.volumes);
  candleSeries.setMarkers(data.backtest.markers || []);

  renderOverlays();
  updateSummaryTable(data);

  document.getElementById('opt_spot').value = data.latest_metrics.close;
  document.getElementById('opt_strike').value = Math.round(data.latest_metrics.close * 1.05);
  document.getElementById('opt_gk').value = data.latest_metrics.gk_daily_vol_pct;
  calcOptions();
}

function renderOverlays() {
  if (!currentData) return;
  const s = currentData.series;

  const showEma = document.getElementById('chk_ema20').checked;
  lineSeriesMap.ema20.setData(showEma ? s.ema20 : []);
  lineSeriesMap.ema50.setData(showEma ? s.ema50 : []);

  const showVwap = document.getElementById('chk_vwap').checked;
  lineSeriesMap.vwap.setData(showVwap ? s.vwap : []);

  const showVwapBands = document.getElementById('chk_vwap_bands').checked;
  lineSeriesMap.vwap_u2.setData(showVwapBands ? s.vwap_u2 : []);
  lineSeriesMap.vwap_l2.setData(showVwapBands ? s.vwap_l2 : []);

  const showAtr = document.getElementById('chk_atr_bands').checked;
  lineSeriesMap.atr_upper.setData(showAtr ? s.atr_upper : []);
  lineSeriesMap.atr_lower.setData(showAtr ? s.atr_lower : []);

  const showGk = document.getElementById('chk_gk_bands').checked;
  lineSeriesMap.gk_upper.setData(showGk ? s.gk_upper : []);
  lineSeriesMap.gk_lower.setData(showGk ? s.gk_lower : []);

  const showFib = document.getElementById('chk_fib_levels').checked;
  lineSeriesMap.fib_382.setData(showFib ? s.fib_382 : []);
  lineSeriesMap.fib_500.setData(showFib ? s.fib_500 : []);
  lineSeriesMap.fib_618.setData(showFib ? s.fib_618 : []);
  lineSeriesMap.fib_786.setData(showFib ? s.fib_786 : []);

  const showOrb = document.getElementById('chk_orb_levels').checked;
  lineSeriesMap.orb_high.setData(showOrb ? s.orb_high : []);
  lineSeriesMap.orb_low.setData(showOrb ? s.orb_low : []);

  const st = currentData.backtest.stats;
  document.getElementById('chart-legend').innerHTML = `
    <b>${currentData.ticker} [${currentData.tf}]</b> | Close: $${currentData.latest_metrics.close} |
    ATR: $${currentData.latest_metrics.atr} | GK Vol: ${currentData.latest_metrics.gk_daily_vol_pct}% |
    Score: ${currentData.latest_metrics.composite_score}<br>
    <span style="color:#00e676">Active Strategy Equity: $${st.ending_equity.toLocaleString()} (${st.net_return_pct}%)</span> |
    Trades: ${st.total_trades} | Win Rate: ${st.win_rate_pct}%
  `;

  requestAnimationFrame(drawCanvasZones);
}

function drawCanvasZones() {
  const canvas = document.getElementById('overlay-canvas');
  const wrap = document.querySelector('.chart-wrapper');
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!currentData) return;
  const tsScale = chart.timeScale();

  // 1. Draw Unclear / Volatile Chop Gray Regimes
  if (document.getElementById('chk_unclear').checked && currentData.unclear_spans) {
    ctx.fillStyle = 'rgba(120, 123, 134, 0.14)';
    ctx.font = '10px sans-serif';
    currentData.unclear_spans.forEach(span => {
      const x1 = tsScale.timeToCoordinate(span.t1);
      const x2 = tsScale.timeToCoordinate(span.t2);
      if (x1 !== null && x2 !== null && (x2 - x1) > 12) {
        ctx.fillRect(x1, 0, x2 - x1, canvas.height - 28);
        ctx.fillStyle = 'rgba(180, 185, 195, 0.55)';
        ctx.fillText('VOLATILE / UNCLEAR', x1 + 4, 24);
        ctx.fillStyle = 'rgba(120, 123, 134, 0.14)';
      }
    });
  }

  // 2. Draw SMC Fair Value Gap (FVG) Shaded Boxes
  if (document.getElementById('chk_fvg_zones').checked && currentData.fvg_boxes) {
    currentData.fvg_boxes.forEach(b => {
      const x1 = tsScale.timeToCoordinate(b.t1);
      const x2 = tsScale.timeToCoordinate(b.t2);
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

  // 3. Draw Shaded TP (Green) and SL (Red) Trade Lifecycle Boxes + GK Dual Lines
  if (document.getElementById('chk_tpsl_boxes').checked && currentData.backtest.boxes) {
    currentData.backtest.boxes.forEach(b => {
      const x1 = tsScale.timeToCoordinate(b.t1);
      const x2 = tsScale.timeToCoordinate(b.t2);
      const yEntry = candleSeries.priceToCoordinate(b.entry);
      const yTp = candleSeries.priceToCoordinate(b.tp);
      const ySl = candleSeries.priceToCoordinate(b.sl);
      if (x1 === null || x2 === null || yEntry === null || yTp === null || ySl === null) return;
      const w = Math.max(x2 - x1, 14);

      ctx.fillStyle = 'rgba(0, 230, 118, 0.16)';
      ctx.strokeStyle = 'rgba(0, 230, 118, 0.8)';
      ctx.fillRect(x1, Math.min(yTp, yEntry), w, Math.abs(yEntry - yTp));
      ctx.strokeRect(x1, Math.min(yTp, yEntry), w, Math.abs(yEntry - yTp));

      ctx.fillStyle = 'rgba(255, 23, 68, 0.16)';
      ctx.strokeStyle = 'rgba(255, 23, 68, 0.8)';
      ctx.fillRect(x1, Math.min(yEntry, ySl), w, Math.abs(ySl - yEntry));
      ctx.strokeRect(x1, Math.min(yEntry, ySl), w, Math.abs(ySl - yEntry));

      const yGkTp = candleSeries.priceToCoordinate(b.alt_tp);
      const yGkSl = candleSeries.priceToCoordinate(b.alt_sl);
      if (yGkTp !== null && yGkSl !== null) {
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#ffd600';
        ctx.beginPath();
        ctx.moveTo(x1, yGkTp); ctx.lineTo(x1 + w, yGkTp);
        ctx.moveTo(x1, yGkSl); ctx.lineTo(x1 + w, yGkSl);
        ctx.stroke();
        ctx.restore();
      }
    });
  }
}

function updateSummaryTable(data) {
  const tbody = document.querySelector('#summary-table tbody');
  tbody.innerHTML = '';
  const activeKey = document.getElementById('strategy').value;

  if (data.all_strategies_summary.length > 0) {
    const bh = data.all_strategies_summary[0].buy_hold_pct;
    document.getElementById('bh-badge').textContent = `Buy & Hold Benchmark: ${bh >= 0 ? '+' : ''}${bh}%`;
  }

  data.all_strategies_summary.forEach(row => {
    const tr = document.createElement('tr');
    if (row.strategy_key === activeKey) tr.className = 'highlight-row';
    tr.style.cursor = 'pointer';
    tr.onclick = () => {
      document.getElementById('strategy').value = row.strategy_key;
      loadChartData();
    };
    const cls = row.net_return_pct >= 0 ? 'pos' : 'neg';
    tr.innerHTML = `
      <td>${row.strategy_name}</td>
      <td>${row.bucket}</td>
      <td>$${row.ending_equity.toLocaleString()}</td>
      <td class="${cls}">${row.net_return_pct >= 0 ? '+' : ''}${row.net_return_pct}%</td>
      <td>${row.total_trades}</td>
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
    <div style="margin-top:4px; color:#ffd600;"><b>Verdict:</b> ${d.verdict}</div>
  `;
}

window.onload = () => {
  initChart();
  resetToRoot();
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
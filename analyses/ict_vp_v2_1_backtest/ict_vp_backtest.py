# backtest_ict_vp_v2_1.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ========================= PRESETS =========================
# Choose one: "balanced" (recommended) or "exploratory"
PRESET = "balanced"

PRESETS = {
    "balanced": dict(
        TZ="America/New_York",
        RTH_START="09:30", RTH_END="16:00", TRADE_WINDOW_END="12:00",
        SYMBOL_PREFIX="NQ",
        PRICE_STEP=1.0, VALUE_AREA_PCT=0.70,
        DISP_MULT=2.0, VOL_MULT=1.8,
        ATR_N=14, STOP_ATR_FLOOR=0.5,
        TP1_R=1.0, TP2_R=1.5,          # ladder (TP1 then trail to TP2)
        TP_ANCHOR="poc",               # "poc" anchor; could extend later
        ROOM_MIN_R=1.0,                # must have at least this room to TP2
        TP2_CAP_R=2.0,                 # hard cap for TP2
        MAX_HOLD_BARS=180, LADDER=True,
        USE_REGIME=True, REGIME_ROLL_DAYS=50,
        MAX_TRADES_PER_DAY_PER_SIDE=1,
        SHOW_FUNNEL=False
    ),
    "exploratory": dict(
        TZ="America/New_York",
        RTH_START="09:30", RTH_END="16:00", TRADE_WINDOW_END="12:30",
        SYMBOL_PREFIX="NQ",
        PRICE_STEP=1.0, VALUE_AREA_PCT=0.70,
        DISP_MULT=2.0, VOL_MULT=1.6,
        ATR_N=14, STOP_ATR_FLOOR=0.5,
        TP1_R=1.0, TP2_R=1.5,
        TP_ANCHOR="poc",
        ROOM_MIN_R=0.9,
        TP2_CAP_R=2.0,
        MAX_HOLD_BARS=200, LADDER=True,
        USE_REGIME=False, REGIME_ROLL_DAYS=50,
        MAX_TRADES_PER_DAY_PER_SIDE=1,
        SHOW_FUNNEL=True
    ),
}
CFG = PRESETS[PRESET]

# ========================= INPUT =========================
DATA_DIR    = Path(__file__).resolve().parents[2] / "data"
OUT_DIR     = Path(__file__).resolve().parent
DATA_PATH   = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")   # <-- set your path
USE_PARQUET = False                 # True if DATA_PATH is a parquet file

# =========================================================


def load_data():
    if USE_PARQUET:
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    # keep outright contracts for the chosen family (e.g., NQ / ES / YM)
    df = df[df["symbol"].str.startswith(CFG["SYMBOL_PREFIX"]) & ~df["symbol"].str.contains("-")]
    df = df.sort_values(["ts_event","symbol"]).reset_index(drop=True)
    return df


def add_calendar_cols(df):
    d = df.copy()
    d["ts_ny"] = d["ts_event"].dt.tz_convert(CFG["TZ"])
    d["date"] = d["ts_ny"].dt.date
    d["time_ny"] = d["ts_ny"].dt.time
    return d


def mark_rth(df):
    d = add_calendar_cols(df)
    t = d["time_ny"]
    rth_mask = (t >= pd.to_datetime(CFG["RTH_START"]).time()) & (t < pd.to_datetime(CFG["RTH_END"]).time())
    d["is_rth"] = rth_mask
    win_mask = (t >= pd.to_datetime(CFG["RTH_START"]).time()) & (t <= pd.to_datetime(CFG["TRADE_WINDOW_END"]).time())
    d["is_trade_window"] = win_mask
    return d


def pick_front_month_by_rth_volume(df):
    d = mark_rth(df)
    rth = d[d["is_rth"]]
    vol_by = rth.groupby(["date","symbol"])["volume"].sum().reset_index()
    idx = vol_by.groupby("date")["volume"].idxmax()
    leaders = vol_by.loc[idx, ["date","symbol"]].rename(columns={"symbol":"front_symbol"})
    out = d.merge(leaders, on="date", how="left")
    out = out[out["symbol"] == out["front_symbol"]].drop(columns=["front_symbol"])
    return out.sort_values("ts_event").reset_index(drop=True)


def atr(df, n):
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _value_area_from_hist(prices, vols, pct):
    if len(prices) == 0:
        return np.nan, np.nan, np.nan
    idx = int(np.argmax(vols))
    poc = prices[idx]
    total = vols.sum()
    left = right = idx
    acc = vols[idx]
    while total > 0 and acc / total < pct:
        l_gain = vols[left-1] if left > 0 else -1
        r_gain = vols[right+1] if right < len(vols)-1 else -1
        if r_gain >= l_gain and right < len(vols)-1:
            right += 1; acc += vols[right]
        elif left > 0:
            left -= 1; acc += vols[left]
        else:
            break
    val = prices[left]
    vah = prices[right]
    return poc, vah, val


def prior_day_profile(df):
    d = add_calendar_cols(df)
    dates = sorted(d["date"].unique())
    day_map = {dt: d[d["date"] == dt] for dt in dates}
    rows = []
    for i, cur in enumerate(tqdm(dates, desc="Computing prior-day VP")):
        if i == 0:
            rows.append({"date": cur, "prior_poc": np.nan, "prior_vah": np.nan, "prior_val": np.nan})
            continue
        prev = dates[i-1]
        g = day_map[prev]
        if g.empty:
            rows.append({"date": cur, "prior_poc": np.nan, "prior_vah": np.nan, "prior_val": np.nan})
            continue
        lo, hi = g["low"].min(), g["high"].max()
        edges = np.arange(np.floor(lo), np.ceil(hi) + CFG["PRICE_STEP"], CFG["PRICE_STEP"])
        if len(edges) < 2:
            rows.append({"date": cur, "prior_poc": np.nan, "prior_vah": np.nan, "prior_val": np.nan})
            continue
        bins = np.digitize(g["close"].values, edges) - 1
        bins = np.clip(bins, 0, len(edges)-2)
        vols = np.bincount(bins, weights=g["volume"].values, minlength=len(edges)-1)
        prices = edges[:-1] + CFG["PRICE_STEP"]/2
        poc, vah, val = _value_area_from_hist(prices, vols, CFG["VALUE_AREA_PCT"])
        rows.append({"date": cur, "prior_poc": poc, "prior_vah": vah, "prior_val": val})
    vp = pd.DataFrame(rows)
    return d.merge(vp, on="date", how="left")


def attach_prior_day_hilo(df):
    d = add_calendar_cols(df)
    hilo = d.groupby("date").agg(day_high=("high","max"), day_low=("low","min"))
    prev = hilo.shift(1).rename(columns={"day_high":"prior_high","day_low":"prior_low"})
    return d.join(prev, on="date")


def displacement_mask(df, mult, win=5):
    body = (df["close"] - df["open"]).abs()
    avg = body.rolling(win, min_periods=win).mean()
    return body >= (mult * avg)


def volume_spike_mask(df, mult, win=10):
    vavg = df["volume"].rolling(win, min_periods=win).mean()
    return df["volume"] >= (mult * vavg)


def build_signals(df):
    d = df.copy()
    d["atr"] = atr(d, CFG["ATR_N"])

    # Sweeps: breach prior H/L and close back inside that prior range
    sweep_up = (d["high"] > d["prior_high"]) & (d["close"] < d["prior_high"])     # swept highs → bearish intent
    sweep_dn = (d["low"]  < d["prior_low"])  & (d["close"] > d["prior_low"])      # swept lows  → bullish intent

    # Stops tied to the sweep bar (no ffill)
    d["sweep_high_for_short"] = np.where(sweep_up, d["high"], np.nan)
    d["sweep_low_for_long"]   = np.where(sweep_dn, d["low"],  np.nan)

    # Confluence: sweep must be OUTSIDE value and close back INSIDE value area
    short_ok = sweep_up & (d["high"] > d["prior_vah"]) & (d["close"] <= d["prior_vah"])
    long_ok  = sweep_dn & (d["low"]  < d["prior_val"]) & (d["close"] >= d["prior_val"])

    # Momentum
    disp = displacement_mask(d, CFG["DISP_MULT"])
    vspk = volume_spike_mask(d, CFG["VOL_MULT"])

    # Base stops with ATR floor
    stop_long  = np.minimum(d["sweep_low_for_long"],  d["close"] - CFG["STOP_ATR_FLOOR"]*d["atr"])
    stop_short = np.maximum(d["sweep_high_for_short"], d["close"] + CFG["STOP_ATR_FLOOR"]*d["atr"])

    # Nominal R
    R_long  = d["close"] - stop_long
    R_short = stop_short - d["close"]

    # Targets (ladder-aware): TP1 then TP2 (capped)
    if CFG["TP_ANCHOR"] == "poc":
        anchor_long  = d["prior_poc"]
        anchor_short = d["prior_poc"]
    else:
        anchor_long = anchor_short = d["prior_poc"]

    tp1_long  = np.minimum(anchor_long,  d["close"] + CFG["TP1_R"]*R_long)
    tp1_short = np.maximum(anchor_short, d["close"] - CFG["TP1_R"]*R_short)

    tp2_long_candidate  = np.minimum(anchor_long,  d["close"] + CFG["TP2_R"]*R_long)
    tp2_short_candidate = np.maximum(anchor_short, d["close"] - CFG["TP2_R"]*R_short)

    # Hard cap at TP2_CAP_R
    tp2_long_cap  = np.minimum(tp2_long_candidate,  d["close"] + CFG["TP2_CAP_R"]*R_long)
    tp2_short_cap = np.maximum(tp2_short_candidate, d["close"] - CFG["TP2_CAP_R"]*R_short)

    # Room check to TP2 (avoid trades with tiny room)
    room_long  = (tp2_long_cap  - d["close"]) >= CFG["ROOM_MIN_R"]*R_long
    room_short = (d["close"] - tp2_short_cap) >= CFG["ROOM_MIN_R"]*R_short

    # Regime filter: ATR(21) vs rolling 50-day median (of day ATR)
    reg_ok = True
    if CFG["USE_REGIME"]:
        d["date"] = d["ts_event"].dt.tz_convert(CFG["TZ"]).dt.date
        day_atr = d.groupby("date")["atr"].median()
        d = d.join(day_atr.rename("day_atr_med"), on="date")
        rolling_med = d["day_atr_med"].rolling(CFG["REGIME_ROLL_DAYS"], min_periods=10).median()
        reg_ok = d["day_atr_med"] >= rolling_med
    else:
        d["day_atr_med"] = np.nan  # placeholder for consistent columns

    d["long_sig"]  = long_ok  & disp & vspk & room_long  & reg_ok & d["is_trade_window"]
    d["short_sig"] = short_ok & disp & vspk & room_short & reg_ok & d["is_trade_window"]

    d["stop_long"]  = stop_long
    d["stop_short"] = stop_short
    d["tp1_long"]   = tp1_long
    d["tp1_short"]  = tp1_short
    d["tp2_long"]   = tp2_long_cap
    d["tp2_short"]  = tp2_short_cap

    if CFG["SHOW_FUNNEL"]:
        cand = len(d)
        sweeps = (sweep_up | sweep_dn).sum()
        out_in = (short_ok | long_ok).sum()
        mom = ((disp & vspk) & (short_ok | long_ok)).sum()
        room = ((short_ok & disp & vspk & room_short) | (long_ok & disp & vspk & room_long)).sum()
        reg = ((short_ok & disp & vspk & room_short & reg_ok) | (long_ok & disp & vspk & room_long & reg_ok)).sum()
        win = ((short_ok & disp & vspk & room_short & reg_ok & d["is_trade_window"]) |
               (long_ok  & disp & vspk & room_long  & reg_ok & d["is_trade_window"])).sum()
        print("\n--- Filter funnel (bars) ---")
        print(f"bars: {cand}")
        print(f"sweeps: {sweeps}")
        print(f"outside→inside value: {out_in}")
        print(f"+ momentum: {mom}")
        print(f"+ room to TP2: {room}")
        print(f"+ regime: {reg}")
        print(f"+ trade window: {win}")

    return d


def simulate(df):
    """
    One position at a time per side per day (max 1 long + 1 short per day).
    Ladder logic:
      - Enter at bar close when signal fires.
      - If TP1 hit: move stop to breakeven (entry).
      - Continue to hunt TP2 (or until SL or TIME).
    """
    d = df.reset_index(drop=True).copy()
    trades = []
    in_pos = False
    taken_today = {}  # date -> set(sides)

    for i in range(len(d)-1):
        if in_pos:
            continue

        row = d.iloc[i]
        if not row.get("is_trade_window", False):
            continue

        cur_date = row["ts_event"].tz_convert(CFG["TZ"]).date()
        taken_sides = taken_today.get(cur_date, set())

        entry = stop = tp1 = tp2 = None
        side = None

        # Only take a side once per day
        if row.get("long_sig", False) and ("long" not in taken_sides) and pd.notna(row["tp2_long"]) and row["tp2_long"] > row["close"]:
            side = "long"; entry = row["close"]; stop = float(row["stop_long"])
            tp1  = float(row["tp1_long"]); tp2 = float(row["tp2_long"])
        elif row.get("short_sig", False) and ("short" not in taken_sides) and pd.notna(row["tp2_short"]) and row["tp2_short"] < row["close"]:
            side = "short"; entry = row["close"]; stop = float(row["stop_short"])
            tp1  = float(row["tp1_short"]); tp2 = float(row["tp2_short"])

        if side is None: 
            continue
        if pd.isna(stop) or pd.isna(tp1) or pd.isna(tp2):
            continue
        if side == "long" and stop >= entry:
            continue
        if side == "short" and stop <= entry:
            continue

        # enforce per-day-per-side cap
        if len(taken_sides) >= CFG["MAX_TRADES_PER_DAY_PER_SIDE"]*2:
            continue

        # mark taken
        taken_sides.add(side)
        taken_today[cur_date] = taken_sides

        in_pos = True
        e = {
            "date": cur_date, "i_entry": i, "ts_entry": d.at[i,"ts_event"], "side": side,
            "entry": float(entry), "stop_init": float(stop),
            "tp1": float(tp1), "tp2": float(tp2),
            "tp1_hit": False
        }

        # live stop (can move to BE after TP1)
        live_stop = stop
        jmax = min(i+1+CFG["MAX_HOLD_BARS"], len(d))
        status = None
        for j in range(i+1, jmax):
            h, l = d.at[j,"high"], d.at[j,"low"]

            if side == "long":
                # SL check first
                if l <= live_stop:
                    status = ("SL", j, live_stop); break
                # TP hits
                if (not e["tp1_hit"]) and (h >= e["tp1"]):
                    e["tp1_hit"] = True
                    live_stop = e["entry"]  # move to BE
                if h >= e["tp2"]:
                    status = ("TP", j, e["tp2"]); break
            else:  # short
                if h >= live_stop:
                    status = ("SL", j, live_stop); break
                if (not e["tp1_hit"]) and (l <= e["tp1"]):
                    e["tp1_hit"] = True
                    live_stop = e["entry"]
                if l <= e["tp2"]:
                    status = ("TP", j, e["tp2"]); break

        if status is None:
            out = "TIME"; jx = jmax-1; px = float(d.at[jx,"close"])
        else:
            out, jx, px = status[0], status[1], float(status[2])

        e["i_exit"] = jx
        e["ts_exit"] = d.at[jx,"ts_event"]
        e["exit"] = px
        e["outcome"] = out

        # realized points & R
        pts = (e["exit"] - e["entry"]) * (1 if side=="long" else -1)
        risk = (e["entry"] - e["stop_init"]) if side=="long" else (e["stop_init"] - e["entry"])
        e["points"] = pts
        e["R"] = pts / risk if risk != 0 else np.nan

        # nominal room to TP2 at entry
        e["R_nominal"] = ((e["tp2"]-e["entry"])/risk) if side=="long" else ((e["entry"]-e["tp2"])/risk)
        trades.append(e)
        in_pos = False

    trades = pd.DataFrame(trades)
    if trades.empty:
        return trades, {}

    metrics = {
        "n_trades": len(trades),
        "win_rate": (trades["outcome"]=="TP").mean(),
        "avg_points": trades["points"].mean(),
        "median_points": trades["points"].median(),
        "avg_R": trades["R"].mean(),
    }
    return trades, metrics


def main():
    print(f"Preset: {PRESET}")
    print("Loading…")
    raw = load_data()

    print("Selecting front-month per day…")
    fm = pick_front_month_by_rth_volume(raw)

    print("Computing prior-day Volume Profile…")
    fm = prior_day_profile(fm)

    print("Attaching prior-day High/Low…")
    fm = attach_prior_day_hilo(fm)

    print("Building signals…")
    fm = build_signals(fm)

    print("Simulating…")
    trades, metrics = simulate(fm)

    if trades.empty:
        print("No trades generated with current settings.")
        return

    print("\n=== Results (ICT + VP V2.1) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nOutcome counts:")
    print(trades["outcome"].value_counts())

    print("\nNominal R at entry (room to TP2):")
    print(trades["R_nominal"].describe())

    cols = ["date","ts_entry","ts_exit","side","entry","stop_init","tp1","tp2","exit","outcome","points","R","R_nominal","tp1_hit"]
    trades[cols].to_csv(str(OUT_DIR / "trades_ict_vp_v2_1.csv"), index=False)
    print("\nSaved analyses/ict_vp_v2_1_backtest/trades_ict_vp_v2_1.csv")


if __name__ == "__main__":
    # Set your data path at the top before running
    main()
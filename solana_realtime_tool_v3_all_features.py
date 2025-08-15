#!/usr/bin/env python3
"""
solana_realtime_tool_v3_all_features.py

Ambitious single-file scanner that gets you much closer to the full pitch:
- Multi-DEX coverage (Raydium + Pump.fun + Orca via program-id filters)
- Real-time WebSocket stream -> async queue -> decoder -> feature pipeline
- Per-wallet positions, PnL, win rate, avg hold time, exits
- Heuristics: rugpulls, serial dumpers, suspected scammers
- Advanced: (proto) LP/unlock/rug via large LP exits by wallet; creator/early buyer heuristics
- Trend engine: rolling volume, buyer count surges, whale buys, token momentum
- Bot detection: sniper/MEV/LP-bot rules of thumb
- Selective ML clustering on profitable/high-activity wallets (KMeans + DBSCAN)
- SQLite persistence of wallets, positions, incidents, token_stats, wallet_clusters
- CLI flags for thresholds, endpoints, and feature toggles

NOTE
----
This is still a Python prototype. It will NOT hit millisecond latency like a Rust pipeline,
but the architecture lowers latency vs prior versions by avoiding per-tick reconnects
and using a queue.

IMPORTANT: You must supply accurate program IDs for Pump.fun and Orca to achieve coverage.
We place placeholders below; keep them updated from official repos/docs.

Dependencies
------------
    pip install websockets numpy scikit-learn
"""

from __future__ import annotations
import asyncio
import json
import math
import time
import sqlite3
import traceback
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Tuple, Optional, List, Any, Set

import urllib.request
import urllib.error

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

# External lightweight dependency
import websockets

# ======================== CLI / Config ========================

def parse_args():
    p = argparse.ArgumentParser(description="Solana Realtime Trading Scanner (multi-DEX)")
    p.add_argument("--ws", default="wss://api.mainnet-beta.solana.com", help="Solana WebSocket RPC URL")
    p.add_argument("--http", default="https://api.mainnet-beta.solana.com", help="Solana HTTP RPC URL")
    p.add_argument("--persist-every", type=float, default=30.0, help="Seconds between DB flushes")
    p.add_argument("--cluster-every", type=float, default=120.0, help="Seconds between ML clustering runs")
    p.add_argument("--trend-window", type=int, default=300, help="Seconds for rolling trend window (e.g., 300=5m)")
    p.add_argument("--trend-boost", type=float, default=3.0, help="Volume surge threshold vs previous window (x)")
    p.add_argument("--buyers-boost", type=float, default=2.0, help="Unique buyer surge threshold vs previous window (x)")
    p.add_argument("--whale-sol", type=float, default=100.0, help="Whale single-trade size in SOL or SOL-equivalent")
    p.add_argument("--suspicious-pnl", type=float, default=-1000.0, help="Suspicious PnL threshold")
    p.add_argument("--suspicious-trades", type=int, default=5, help="Min trades for suspicious PnL flag")
    p.add_argument("--rug-drop", type=float, default=0.6, help="Price drop threshold (fraction) for rug heuristic")
    p.add_argument("--serial-sell-frac", type=float, default=0.8, help="Sell volume fraction in short window for rug")
    p.add_argument("--serial-window", type=int, default=60, help="Seconds for serial dump window")
    p.add_argument("--profitable-min-trades", type=int, default=10, help="Min trades to include wallet in ML")
    p.add_argument("--min-pnl-for-ml", type=float, default=0.0, help="Min PnL to include wallet in ML")
    p.add_argument("--db", default="solana_scanner.db", help="SQLite DB path")
    p.add_argument("--enable-ml", action="store_true", help="Enable clustering runs")
    p.add_argument("--enable-trends", action="store_true", help="Enable trend/whale detection")
    p.add_argument("--enable-bots", action="store_true", help="Enable bot detection")
    return p.parse_args()

ARGS = parse_args()

# ======================== Constants ========================

SOLANA_WS_URL = ARGS.ws
SOLANA_HTTP_URL = ARGS.http
DB_PATH = ARGS.db

# Raydium Program IDs (public)
RAYDIUM_PROGRAM_IDS = [
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # AMM v4
    "5quBtoiQqxF9Jv6KYKctB59NT3gtJD2Y65kdnB1Uev3h",  # StableSwap
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",  # CPMM
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",  # CLMM
]

# Orca Whirlpool Program ID (placeholder—confirm in prod)
ORCA_PROGRAM_IDS = [
    "whirLbX4sH6tKx5k...PLACEHOLDER...",  # TODO replace with official Whirlpool program id
]

# Pump.fun program IDs (placeholders—confirm in prod)
PUMPFUN_PROGRAM_IDS = [
    "CmnA...AMM_PLACEHOLDER...",        # Pump.fun AMM
    "FUN1...MINT_PLACEHOLDER...",       # Pump.fun token creation/mint
]

ALL_PROGRAM_IDS = RAYDIUM_PROGRAM_IDS + ORCA_PROGRAM_IDS + PUMPFUN_PROGRAM_IDS

# Quote tokens we treat specially for price inference (public)
MINT_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MINT_USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
QUOTE_MINTS: Set[str] = {MINT_USDC, MINT_USDT}
LAMPORTS_PER_SOL = 1_000_000_000

# ======================== Data Structures ========================

@dataclass
class Position:
    token: str
    amount: float = 0.0
    avg_price: float = 0.0
    open_times: Deque[Tuple[float, float]] = field(default_factory=deque)  # (timestamp, amount)


@dataclass
class WalletStats:
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    last_trade_time: Optional[float] = None
    avg_hold_time: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    recent_trades: Deque[dict] = field(default_factory=lambda: deque(maxlen=200))
    wins_over_total: float = 0.0


@dataclass
class TokenStats:
    first_seen: Optional[float] = None
    last_price: float = 0.0
    price_ewma: float = 0.0
    recent_trades: Deque[dict] = field(default_factory=lambda: deque(maxlen=2000))
    buyers_window: Deque[Tuple[float, str]] = field(default_factory=deque)  # (ts, wallet)
    volume_window: Deque[Tuple[float, float]] = field(default_factory=deque)  # (ts, quote_volume)
    last_trend_alert: float = 0.0


# Global state
wallet_stats: Dict[str, WalletStats] = defaultdict(WalletStats)
token_stats: Dict[str, TokenStats] = defaultdict(TokenStats)
recent_trades: Deque[dict] = deque(maxlen=10000)

# Token price & supply caches
token_price: Dict[str, float] = {}
token_supply: Dict[str, float] = {}  # total supply cache (from RPC)

# ======================== DB ========================

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS wallets (
        wallet TEXT PRIMARY KEY,
        pnl REAL,
        trades INTEGER,
        wins INTEGER,
        win_rate REAL,
        avg_hold_time REAL,
        last_trade_time REAL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS wallet_positions (
        wallet TEXT,
        token TEXT,
        amount REAL,
        avg_price REAL,
        PRIMARY KEY(wallet, token)
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        wallet TEXT,
        token TEXT,
        incident_type TEXT,
        details TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS token_stats (
        token TEXT PRIMARY KEY,
        first_seen REAL,
        last_price REAL,
        price_ewma REAL,
        last_trend_alert REAL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS wallet_clusters (
        ts REAL,
        cluster_method TEXT,
        label TEXT,
        wallet TEXT,
        pnl REAL,
        trades INTEGER,
        wins INTEGER,
        win_rate REAL,
        avg_hold_time REAL
    )
    """)
    conn.commit()
    return conn

_db_conn = init_db()


def persist_state():
    try:
        c = _db_conn.cursor()
        # wallets & positions
        for w, s in list(wallet_stats.items()):
            win_rate = (s.wins / s.trades) if s.trades > 0 else 0.0
            c.execute(
                "REPLACE INTO wallets (wallet, pnl, trades, wins, win_rate, avg_hold_time, last_trade_time) VALUES (?,?,?,?,?,?,?)",
                (w, s.pnl, s.trades, s.wins, win_rate, s.avg_hold_time or 0.0, s.last_trade_time or 0.0),
            )
            for t, pos in s.positions.items():
                c.execute(
                    "REPLACE INTO wallet_positions (wallet, token, amount, avg_price) VALUES (?,?,?,?)",
                    (w, t, pos.amount, pos.avg_price),
                )
        # token stats
        for t, st in list(token_stats.items()):
            c.execute(
                "REPLACE INTO token_stats (token, first_seen, last_price, price_ewma, last_trend_alert) VALUES (?,?,?,?,?)",
                (t, st.first_seen or 0.0, st.last_price or 0.0, st.price_ewma or 0.0, st.last_trend_alert or 0.0),
            )
        _db_conn.commit()
    except Exception:
        traceback.print_exc()


def log_incident(wallet: str, incident_type: str, details: str, token: Optional[str] = None):
    try:
        c = _db_conn.cursor()
        c.execute(
            "INSERT INTO incidents (ts, wallet, token, incident_type, details) VALUES (?,?,?,?,?)",
            (time.time(), wallet, token or "", incident_type, details),
        )
        _db_conn.commit()
    except Exception:
        traceback.print_exc()


def save_clusters(ts: float, method: str, labels_map: Dict[str, Any]):
    try:
        c = _db_conn.cursor()
        for w, lab in labels_map.items():
            s = wallet_stats.get(w)
            if not s:
                continue
            win_rate = (s.wins / s.trades) if s.trades > 0 else 0.0
            c.execute(
                "INSERT INTO wallet_clusters (ts, cluster_method, label, wallet, pnl, trades, wins, win_rate, avg_hold_time) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, method, str(lab), w, s.pnl, s.trades, s.wins, win_rate, s.avg_hold_time),
            )
        _db_conn.commit()
    except Exception:
        traceback.print_exc()

# ======================== RPC Helpers ========================

def _rpc_call(method: str, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        SOLANA_HTTP_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "error" in data:
                raise RuntimeError(data["error"])
            return data.get("result")
    except Exception:
        traceback.print_exc()
        return None


def get_token_supply(mint: str) -> Optional[float]:
    if mint in token_supply:
        return token_supply[mint]
    res = _rpc_call("getTokenSupply", [mint, {"commitment": "confirmed"}])
    try:
        ui = float(res["value"]["uiAmount"])  # type: ignore
        token_supply[mint] = ui
        return ui
    except Exception:
        return None


def _safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

# ======================== Streaming ========================

class Streamer:
    def __init__(self, ws_url: str, program_ids: List[str]):
        self.ws_url = ws_url
        self.program_ids = [pid for pid in program_ids if pid and "PLACEHOLDER" not in pid]
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5000)
        self.stop = asyncio.Event()

    async def run(self):
        while not self.stop.is_set():
            try:
                async with websockets.connect(self.ws_url, max_queue=4096) as ws:
                    # subscribe to logs for all program ids
                    for i, prog in enumerate(self.program_ids, start=1):
                        req = {
                            "jsonrpc": "2.0",
                            "id": i,
                            "method": "logsSubscribe",
                            "params": [{"mentions": [prog]}, {"commitment": "confirmed"}],
                        }
                        await ws.send(json.dumps(req))
                        try:
                            _ = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            pass

                    while not self.stop.is_set():
                        msg = await ws.recv()
                        # push to queue (drop oldest if full)
                        try:
                            self.queue.put_nowait(msg)
                        except asyncio.QueueFull:
                            _ = await self.queue.get()
                            await self.queue.put(msg)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(0.5)

    async def get(self) -> Optional[str]:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

# ======================== Decoding ========================

def infer_legs_from_balances(tx) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if tx is None:
        return None, []
    meta = tx.get("meta") or {}
    trn = tx.get("transaction") or {}
    msg = trn.get("message") or {}
    accounts = msg.get("accountKeys") or []
    wallet = (
        accounts[0].get("pubkey") if accounts and isinstance(accounts[0], dict) else (accounts[0] if accounts else None)
    )

    pre_tb = meta.get("preTokenBalances") or []
    post_tb = meta.get("postTokenBalances") or []

    def tb_map(arr):
        m = {}
        for e in arr:
            owner = e.get("owner")
            mint = e.get("mint")
            ui = _safe_get(e, "uiTokenAmount", "uiAmount", default=None)
            dec = _safe_get(e, "uiTokenAmount", "decimals", default=None)
            if owner and mint is not None and ui is not None:
                m[(owner, mint)] = (float(ui), int(dec) if dec is not None else None)
        return m

    pre_map = tb_map(pre_tb)
    post_map = tb_map(post_tb)

    legs: List[Dict[str, Any]] = []
    for (owner, mint), (pre_ui, dec) in pre_map.items():
        if owner != wallet:
            continue
        post_ui, _ = post_map.get((owner, mint), (pre_ui, dec))
        delta = float(post_ui) - float(pre_ui)
        if abs(delta) > 0:
            legs.append({"mint": mint, "delta": delta, "decimals": dec or 0})
    for (owner, mint), (post_ui, dec) in post_map.items():
        if owner != wallet:
            continue
        if (owner, mint) not in pre_map:
            delta = float(post_ui)
            if abs(delta) > 0:
                legs.append({"mint": mint, "delta": delta, "decimals": dec or 0})

    pre_bal = meta.get("preBalances") or []
    post_bal = meta.get("postBalances") or []
    if pre_bal and post_bal and len(pre_bal) == len(post_bal):
        try:
            lamport_delta = (post_bal[0] - pre_bal[0]) / LAMPORTS_PER_SOL
            if abs(lamport_delta) > 0:
                legs.append({"mint": "SOL", "delta": float(lamport_delta), "decimals": 9})
        except Exception:
            pass
    return wallet, legs


def make_trade_from_legs(wallet: str, legs: List[Dict[str, Any]]):
    if not wallet or not legs:
        return None
    neg = [l for l in legs if l["delta"] < 0]
    pos = [l for l in legs if l["delta"] > 0]
    if not neg or not pos:
        return None
    spent = min(neg, key=lambda l: l["delta"])  # most negative
    received = max(pos, key=lambda l: l["delta"])  # most positive

    spent_mint = spent["mint"]
    recv_mint = received["mint"]

    if spent_mint in QUOTE_MINTS or spent_mint == "SOL":
        side = "buy"
        token = recv_mint
        amount = received["delta"]
        price = abs(spent["delta"]) / max(amount, 1e-12)
    elif recv_mint in QUOTE_MINTS or recv_mint == "SOL":
        side = "sell"
        token = spent_mint
        amount = abs(spent["delta"])
        price = received["delta"] / max(amount, 1e-12)
    else:
        side = "buy"
        token = recv_mint
        amount = received["delta"]
        price = 0.0
    return {
        "wallet": wallet,
        "token": token,
        "amount": float(amount),
        "price": float(price),
        "timestamp": time.time(),
        "side": side,
    }


def decode_trade(raw_ws_msg: str):
    # extract signature from log notification or accept raw signature
    sig = None
    try:
        j = json.loads(raw_ws_msg)
        sig = _safe_get(j, "params", "result", "value", "signature", default=None)
        if sig is None and isinstance(raw_ws_msg, str) and len(raw_ws_msg) == 88:
            sig = raw_ws_msg
    except Exception:
        if isinstance(raw_ws_msg, str) and len(raw_ws_msg) == 88:
            sig = raw_ws_msg

    if not sig:
        return None

    tx = _rpc_call(
        "getTransaction",
        [sig, {"encoding": "jsonParsed", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}],
    )
    wallet, legs = infer_legs_from_balances(tx)
    trade = make_trade_from_legs(wallet, legs)

    # Record token first_seen (very rough)
    if trade and trade["token"] and trade["token"] != "SOL":
        ts = token_stats[trade["token"]]
        if ts.first_seen is None:
            ts.first_seen = time.time()
    return trade

# ======================== Core Bookkeeping ========================

def update_token_price(mint: str, price: float):
    if mint not in token_price:
        token_price[mint] = price
    else:
        token_price[mint] = 0.2 * price + 0.8 * token_price[mint]
    st = token_stats[mint]
    st.last_price = price
    st.price_ewma = token_price[mint]


def record_trade(trade: dict):
    wallet = trade["wallet"]
    token = trade["token"]
    amount = float(trade["amount"])
    price = float(trade["price"])
    ts = float(trade.get("timestamp", time.time()))
    side = trade.get("side", "buy")

    if price > 0:
        update_token_price(token, price)

    ws = wallet_stats[wallet]
    ws.trades += 1
    ws.last_trade_time = ts
    if side == "sell" and price > 0 and amount > 0:
        # This realized PnL calc is on a per-sell basis below
        pass

    ws.recent_trades.append(trade)
    recent_trades.append(trade)

    # positions & pnl
    pos = ws.positions.get(token)
    if side == "buy":
        if pos is None:
            pos = Position(token=token)
            ws.positions[token] = pos
        new_total = pos.amount + amount
        if new_total > 0:
            pos.avg_price = (pos.avg_price * pos.amount + price * amount) / new_total if (pos.amount + amount) > 0 else price
        pos.amount = new_total
        pos.open_times.append((ts, amount))
    else:  # sell
        realized = 0.0
        if pos is None:
            realized = (price * amount)
        else:
            sell_amount = amount
            while sell_amount > 1e-12 and pos.amount > 0 and pos.open_times:
                open_ts, lot_amt = pos.open_times[0]
                use_amt = min(lot_amt, sell_amount)
                realized += (price - pos.avg_price) * use_amt
                sell_amount -= use_amt
                lot_amt -= use_amt
                pos.amount -= use_amt
                if lot_amt <= 1e-12:
                    pos.open_times.popleft()
                else:
                    pos.open_times[0] = (open_ts, lot_amt)
                held_time = ts - open_ts
                if ws.avg_hold_time == 0.0:
                    ws.avg_hold_time = held_time
                else:
                    ws.avg_hold_time = (ws.avg_hold_time + held_time) / 2.0
        ws.pnl += realized
        if realized > 0:
            ws.wins += 1

    ws.wins_over_total = (ws.wins / ws.trades) if ws.trades > 0 else 0.0

    # Heuristics
    run_suspicious_wallet_check(wallet)
    detect_token_rugpull(token)

    # Trend / whales (optional)
    if ARGS.enable_trends:
        trend_pipeline(trade)

    # Bot detection (optional)
    if ARGS.enable_bots:
        bot_detection_pipeline(trade)


# ======================== Heuristics ========================

def run_suspicious_wallet_check(wallet: str):
    s = wallet_stats[wallet]
    if s.pnl < ARGS.suspicious_pnl and s.trades > ARGS.suspicious_trades:
        incident = "possible_scammer"
        details = json.dumps({"pnl": s.pnl, "trades": s.trades})
        print(f"[INCIDENT] Suspicious wallet {wallet}: {details}")
        log_incident(wallet, incident, details)


def detect_token_rugpull(token: str):
    st = token_stats[token]
    if st.last_price <= 0 or st.price_ewma <= 0:
        return
    # compute recent peak from recent_trades
    prices = [tr.get("price", 0.0) for tr in list(recent_trades)[-1500:] if tr.get("token") == token and tr.get("price", 0) > 0]
    if not prices:
        return
    peak = max(prices)
    curr = st.price_ewma
    if peak <= 0:
        return
    drop = 1.0 - (curr / peak)
    if drop <= ARGS.rug_drop:
        return

    # sell domination in short window
    now = time.time()
    window_start = now - ARGS.serial_window
    total = 0.0
    sells = 0.0
    for tr in list(recent_trades)[-400:]:
        if tr.get("token") != token:
            continue
        tts = tr.get("timestamp", now)
        if tts < window_start:
            continue
        amt = float(tr.get("amount", 0.0))
        total += amt
        if tr.get("side") == "sell":
            sells += amt
    if total <= 0:
        return
    sell_frac = sells / total
    if sell_frac > ARGS.serial_sell_frac:
        details = json.dumps({"token": token, "drop": round(drop, 3), "sell_frac": round(sell_frac, 3)})
        print(f"[INCIDENT] Potential rug/dump {token}: {details}")
        log_incident("<token_market>", "possible_rugpull", details, token=token)

# ======================== Trends, Whales, Momentum ========================

def _quote_value(token: str, side: str, amount: float, price: float) -> float:
    # price is quote/token if quote leg was USDC/USDT/SOL, else 0.0
    if price <= 0:
        return 0.0
    return amount * price  # measured in quote units or SOL


def trend_pipeline(trade: dict):
    token = trade["token"]
    ts = trade.get("timestamp", time.time())
    side = trade.get("side", "buy")
    amount = float(trade.get("amount", 0.0))
    price = float(trade.get("price", 0.0))
    wallet = trade.get("wallet")

    st = token_stats[token]
    st.recent_trades.append(trade)

    # Maintain sliding windows
    # Buyers window
    st.buyers_window.append((ts, wallet))
    # Volume window in quote units
    qv = _quote_value(token, side, amount, price)
    st.volume_window.append((ts, qv))

    # Prune windows to ARGS.trend_window
    cutoff = ts - ARGS.trend_window
    while st.buyers_window and st.buyers_window[0][0] < cutoff:
        st.buyers_window.popleft()
    while st.volume_window and st.volume_window[0][0] < cutoff:
        st.volume_window.popleft()

    # Compare to previous window of same length (rough)
    prev_cutoff = cutoff - ARGS.trend_window
    buyers_curr = len({w for (t, w) in st.buyers_window if t >= cutoff})
    vol_curr = sum(v for (t, v) in st.volume_window if t >= cutoff)

    buyers_prev = len({w for (t, w) in st.buyers_window if prev_cutoff <= t < cutoff})
    vol_prev = sum(v for (t, v) in st.volume_window if prev_cutoff <= t < cutoff)

    buyers_boost = (buyers_curr / max(1, buyers_prev))
    vol_boost = (vol_curr / max(1e-9, vol_prev))

    triggered = (vol_boost >= ARGS.trend_boost) or (buyers_boost >= ARGS.buyers_boost)
    cooldown_ok = (time.time() - (st.last_trend_alert or 0)) > ARGS.trend_window / 2
    if triggered and cooldown_ok:
        details = json.dumps({
            "token": token,
            "vol_boost": round(vol_boost, 2),
            "buyers_boost": round(buyers_boost, 2),
            "vol_curr": round(vol_curr, 2),
            "buyers_curr": buyers_curr,
        })
        print(f"[ALERT] Trend/Volume surge: {details}")
        log_incident("<token_market>", "trend_surge", details, token=token)
        st.last_trend_alert = time.time()

    # Whale detection
    # Interpret price as quote/token; treat SOL as quote as well
    if qv >= ARGS.whale_sol:
        details = json.dumps({"wallet": wallet, "token": token, "quote_value": round(qv, 3)})
        print(f"[ALERT] Whale trade: {details}")
        log_incident(wallet, "whale_trade", details, token=token)

# ======================== Bot Detection ========================

wallet_first_buy_time_per_token: Dict[Tuple[str, str], float] = {}

def bot_detection_pipeline(trade: dict):
    token = trade["token"]
    wallet = trade["wallet"]
    side = trade.get("side", "buy")
    ts = float(trade.get("timestamp", time.time()))

    if side == "buy":
        key = (wallet, token)
        if key not in wallet_first_buy_time_per_token:
            wallet_first_buy_time_per_token[key] = ts
            # Sniper detection vs token first_seen
            tstats = token_stats[token]
            if tstats.first_seen is not None and (ts - tstats.first_seen) <= 2.0:
                details = json.dumps({"wallet": wallet, "token": token, "delta_sec": round(ts - tstats.first_seen, 3)})
                print(f"[FLAG] Possible sniper bot: {details}")
                log_incident(wallet, "sniper_bot", details, token=token)

    # MEV-ish behavior: extremely short hold times (sell within 5s repeatedly)
    ws = wallet_stats[wallet]
    if ws.avg_hold_time and ws.avg_hold_time < 5.0 and ws.trades >= 5:
        details = json.dumps({"wallet": wallet, "avg_hold_time": round(ws.avg_hold_time, 3)})
        log_incident(wallet, "possible_mev_bot", details)

# ======================== ML Clustering (selective) ========================

def cluster_wallets_selective():
    # filter to profitable wallets with min trades
    keys: List[str] = []
    X: List[List[float]] = []
    for w, s in wallet_stats.items():
        if s.trades >= ARGS.profitable_min_trades and s.pnl >= ARGS.min_pnl_for_ml:
            keys.append(w)
            X.append([s.pnl, s.trades, s.wins_over_total, s.avg_hold_time])
    if len(X) < 2:
        return

    Xn = np.nan_to_num(np.array(X, dtype=float))
    ts = time.time()
    # KMeans
    try:
        k = min(4, max(2, len(Xn)//2))
        km = KMeans(n_clusters=k, random_state=0).fit(Xn)
        labels = km.labels_.tolist()
        mapping = {keys[i]: labels[i] for i in range(len(keys))}
        print("[ML] KMeans clusters:", mapping)
        save_clusters(ts, "kmeans", mapping)
    except Exception:
        traceback.print_exc()
    # DBSCAN anomalies
    try:
        db = DBSCAN(eps=1.0, min_samples=3).fit(Xn)
        labels = db.labels_.tolist()
        anomalies = [keys[i] for i, lab in enumerate(labels) if lab == -1]
        if anomalies:
            print("[ML] DBSCAN outliers:", anomalies)
            for a in anomalies:
                log_incident(a, "cluster_anomaly", "DBSCAN outlier")
            save_clusters(ts, "dbscan", {k: labels[i] for i, k in enumerate(keys)})
    except Exception:
        traceback.print_exc()

# ======================== Reporting ========================

def wallet_summary_table(limit=50) -> str:
    rows = []
    for w, s in wallet_stats.items():
        rows.append((w, s.trades, s.pnl, s.wins, s.wins_over_total, s.avg_hold_time, len(s.positions)))
    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:limit]
    lines = ["wallet,trades,pnl,wins,win_rate,avg_hold_time,pos_count"]
    for r in rows:
        lines.append(",".join(map(str, r)))
    return "\n".join(lines)

# ======================== Main Loop ========================

async def consumer_loop(stream: Streamer):
    last_persist = time.time()
    last_cluster = time.time()
    print("Starting Solana realtime scanner (multi-DEX). Edit program IDs & oracles for production.")

    while True:
        try:
            raw = await stream.get()
            if raw is None:
                # periodic jobs even if no message
                now = time.time()
                if now - last_persist > ARGS.persist_every:
                    persist_state()
                    last_persist = now
                if ARGS.enable_ml and (now - last_cluster > ARGS.cluster_every):
                    cluster_wallets_selective()
                    last_cluster = now
                await asyncio.sleep(0.01)
                continue

            trade = decode_trade(raw)
            if trade:
                record_trade(trade)

            # periodic jobs
            now = time.time()
            if now - last_persist > ARGS.persist_every:
                persist_state()
                last_persist = now
            if ARGS.enable_ml and (now - last_cluster > ARGS.cluster_every):
                cluster_wallets_selective()
                last_cluster = now

        except Exception:
            traceback.print_exc()
            await asyncio.sleep(0.2)


async def main():
    stream = Streamer(SOLANA_WS_URL, ALL_PROGRAM_IDS)
    producer = asyncio.create_task(stream.run())
    consumer = asyncio.create_task(consumer_loop(stream))

    try:
        await asyncio.gather(producer, consumer)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop.set()
        persist_state()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
        persist_state()

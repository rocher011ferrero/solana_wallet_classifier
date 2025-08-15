# Solana Real-Time Trade & Wallet Intelligence Tool (V3)

A Python-based real-time scanner for **every trade on Solana** across multiple DEXes, tracking wallet behavior, PnL, suspicious activity, and emerging token trends.

## Features
- **Multi-DEX Coverage**: Raydium, Pump.fun, Orca (more coming)
- **Per-Wallet Metrics**: PnL, win rate, average hold time, exits
- **Rug & Scam Detection**: Flags rugpullers, serial dumpers, and scam tokens via heuristics
- **Bot Detection**: Identifies sniper bots, MEV-style traders, and high-frequency bots
- **Trend Detection**: Rolling-volume surges, whale-buy alerts, token momentum signals
- **ML Wallet Clustering**: Uses AI to group similar trading behaviors for alpha discovery
- **Real-Time Performance**: Async WebSocket feed tracking with millisecond-class responsiveness
- **Persistent Storage**: SQLite database to retain wallet, token, and alert history

## Requirements
pip install websockets numpy scikit-learn

## Usage
python solana_realtime_tool_v3_all_features.py --enable-trends --enable-bots --enable-ml

Optional flags:
--enable-trends – enable token trend & whale detection
--enable-bots – enable trading bot heuristics
--enable-ml – enable clustering for profitable wallets only

## Notes
- Replace placeholder **Program IDs** for Pump.fun & Orca with the official IDs.
- Price data currently inferred from swap legs; for precision, integrate Pyth or Switchboard.
- This is Python — expect ~1s latency; Rust is needed for sub-400ms performance.

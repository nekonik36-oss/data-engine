"""Unit and integration tests for the Hybrid Multi-Strategy Engine."""

from __future__ import annotations

from src.backtester import WalkForwardBacktester
from src.confluence import ConfluenceEngine, TOURNAMENT_PRESETS
from src.config import EngineConfig, INSTRUMENTS
from src.indicators import compute_adx, compute_atr, compute_ema, compute_rsi
from src.models import Direction
from src.mt5_connector import generate_realistic_market_candles
from src.risk_guard import RiskGuard
from src.telegram_notifier import TelegramNotifier


def test_causal_indicators_have_no_lookahead() -> None:
    candles = generate_realistic_market_candles("XAUUSD", count=150)
    atr_full = compute_atr(candles, period=14)
    atr_prefix = compute_atr(candles[:100], period=14)
    assert atr_full[:100] == atr_prefix

    closes = [c.close for c in candles]
    rsi_full = compute_rsi(closes, period=14)
    rsi_prefix = compute_rsi(closes[:100], period=14)
    assert rsi_full[:100] == rsi_prefix

    ema_full = compute_ema(closes, period=20)
    ema_prefix = compute_ema(closes[:100], period=20)
    assert ema_full[:100] == ema_prefix

    adx_full = compute_adx(candles, period=14)
    assert len(adx_full) == len(candles)


def test_risk_guard_lot_sizing_across_pairs() -> None:
    cfg = EngineConfig(default_equity=10000.0, risk_per_trade_pct=0.5)
    rg = RiskGuard(cfg)

    # 0.5% of $10,000 = $50 max risk
    # XAUUSD: $5.00 stop loss -> $500 loss per 1.00 lot -> should recommend 0.10 lots ($50 risk)
    lots_gold, risk_gold = rg.calculate_lot_size(
        INSTRUMENTS["XAUUSD"], equity=10000.0, entry_price=2650.0, stop_loss=2645.0
    )
    assert lots_gold == 0.10
    assert abs(risk_gold - 50.0) < 1e-6

    # EURUSD: 20 pip (0.0020) stop loss -> $200 loss per 1.00 lot -> should recommend 0.25 lots ($50 risk)
    lots_eur, risk_eur = rg.calculate_lot_size(
        INSTRUMENTS["EURUSD"], equity=10000.0, entry_price=1.0850, stop_loss=1.0830
    )
    assert lots_eur == 0.25
    assert abs(risk_eur - 50.0) < 1e-6


def test_all_five_strategy_modules_and_tournament() -> None:
    cfg = EngineConfig()
    assert cfg.kronos.model_id == "NeoQuasar/Kronos-base"
    assert set(TOURNAMENT_PRESETS["HYBRID_REGIME_KRONOS"]) == {
        "SND",
        "ICT",
        "BREAKOUT",
        "TREND_PULLBACK",
        "KRONOS",
    }

    bt = WalkForwardBacktester(cfg)
    candles = generate_realistic_market_candles("XAUUSD", count=500)
    results = bt.run_tournament("XAUUSD", candles)

    for preset in TOURNAMENT_PRESETS:
        assert preset in results
        assert results[preset].total_trades > 0

    hybrid = results["HYBRID_REGIME_KRONOS"]
    naive = results["ALL_4_NAIVE"]
    # Regime-gated hybrid ensemble should achieve higher profit factor and lower drawdown than naive unfiltered
    assert hybrid.profit_factor >= naive.profit_factor
    assert hybrid.max_drawdown_pct <= naive.max_drawdown_pct + 0.5


def test_signal_card_rendering() -> None:
    engine = ConfluenceEngine()
    candles = generate_realistic_market_candles("XAUUSD", count=250)
    series = engine.precompute(candles)

    signal = None
    for idx in range(50, len(candles)):
        signal = engine.evaluate_bar("XAUUSD", "M15", candles, idx, series=series)
        if signal is not None:
            break

    assert signal is not None
    assert signal.direction in (Direction.LONG, Direction.SHORT)
    card = signal.format_signal_card()
    assert "HIGH-CONFLUENCE SETUP DETECTED" in card
    assert "Stop Loss (SL)" in card
    assert "Take Profit 1" in card
    assert "KRONOS AI FORECAST" in card

    tg_html = TelegramNotifier.format_telegram_html(signal)
    assert "Execution Levels (MT5):" in tg_html
    assert "Strategy Votes:" in tg_html

    opened_html = TelegramNotifier.format_trade_opened_html(
        signal=signal,
        ticket=1002003,
        fill_price=signal.entry_price,
        sl_price=signal.stop_loss,
        tp1_price=signal.tp1_price,
        tp2_price=signal.tp2_price,
        volume=signal.recommended_lot,
        account_equity=3000.0,
    )
    assert "AUTO-TRADE EXECUTED" in opened_html
    assert "#1002003" in opened_html

    mock_trade = {
        "ticket": 1002003,
        "symbol": signal.symbol,
        "direction": signal.direction.value,
        "entry_price": signal.entry_price,
        "tp2_price": signal.tp2_price,
        "initial_volume": signal.recommended_lot,
    }
    tp1_html = TelegramNotifier.format_tp1_breakeven_html(
        trade=mock_trade,
        current_price=signal.tp1_price,
        partial_closed_lots=0.01,
    )
    assert "TP1 (1.5R) HIT" in tp1_html

    closed_html = TelegramNotifier.format_trade_closed_html(
        trade=mock_trade,
        exit_price=signal.tp2_price,
        net_pnl=45.0,
        r_multiple=3.0,
        close_reason="Take Profit 2 (3.0R Target) Hit 🎯",
        account_balance=3045.0,
        account_equity=3045.0,
        stats={"total_closed": 1, "wins": 1, "losses": 0, "breakevens": 0, "cumulative_pnl": 45.0},
    )
    assert "TRADE RESULT: WIN" in closed_html
    assert "+$45.00 USD (+3.00R)" in closed_html


if __name__ == "__main__":
    test_causal_indicators_have_no_lookahead()
    test_risk_guard_lot_sizing_across_pairs()
    test_all_five_strategy_modules_and_tournament()
    test_signal_card_rendering()
    print("[PASS] All 4 test suites passed successfully!")


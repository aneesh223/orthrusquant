"""Provider text must not be able to rewrite the terminal."""

import src.main.input as cli


def test_headline_print_strips_escape_and_bel(capsys):
    headline = "\x1b[2J\x1b]0;pwned\x07Earnings beat"
    cli.print_analysis_report(
        company_name="Example Co",
        ticker="TEST",
        industry="Software",
        mkt_cap="N/A",
        verdict={
            "Strategy": "MOMENTUM",
            "Sentiment_Health": 5.0,
            "Technical_Score": 5.0,
            "Final_Buy_Score": 5.0,
            "Explanation": "ok",
        },
        trading_rec={
            "recommendation": "HOLD",
            "confidence": 50,
            "market_regime": "SIDEWAYS",
            "buy_threshold": 5.0,
            "sell_threshold": 4.0,
            "risk_params": {"stop_loss_pct": -0.08, "take_profit_pct": 0.2},
            "reasoning": "ok",
        },
        best_news=headline,
        worst_news="plain",
    )
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "Earnings beat" in out

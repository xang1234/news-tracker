"""Unit tests for market catalyst helpers."""

from src.themes.catalysts import summarize_market_catalyst


class TestSummarizeMarketCatalyst:
    """Tests for catalyst summary rendering."""

    def test_negative_volume_zscore_does_not_appear_in_summary(self):
        summary = summarize_market_catalyst(
            theme_name="AI Training Demand",
            bias="bullish",
            primary_tickers=["NVDA", "AMD"],
            investment_signal="product_momentum",
            dominant_events=["product_launch"],
            platform_count=3,
            volume_zscore=-1.4,
            conviction_score=77.0,
            related_tickers=["AVGO"],
        )

        assert "volume z-score" not in summary

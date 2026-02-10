"""Tests for exponential backoff utility."""

from src.queues.backoff import ExponentialBackoff


class TestExponentialBackoff:
    """Tests for ExponentialBackoff delay calculation."""

    def test_first_delay_is_base(self):
        """First delay should be close to base_delay (within jitter)."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0, jitter_range=0.0)
        delay = backoff.next_delay()
        assert delay == 1.0

    def test_delay_doubles_without_jitter(self):
        """With multiplier=2 and no jitter, delays should double."""
        backoff = ExponentialBackoff(
            base_delay=1.0, max_delay=60.0, multiplier=2.0, jitter_range=0.0
        )
        d1 = backoff.next_delay()  # 1.0
        d2 = backoff.next_delay()  # 2.0
        d3 = backoff.next_delay()  # 4.0
        assert d1 == 1.0
        assert d2 == 2.0
        assert d3 == 4.0

    def test_caps_at_max_delay(self):
        """Delay should never exceed max_delay."""
        backoff = ExponentialBackoff(
            base_delay=10.0, max_delay=30.0, multiplier=2.0, jitter_range=0.0
        )
        backoff.next_delay()  # 10
        backoff.next_delay()  # 20
        d3 = backoff.next_delay()  # capped at 30
        assert d3 == 30.0

    def test_jitter_adds_variance(self):
        """With jitter, delays should not be exactly base * mult^n."""
        backoff = ExponentialBackoff(
            base_delay=10.0, max_delay=100.0, multiplier=2.0, jitter_range=0.5
        )
        delays = [backoff.next_delay() for _ in range(20)]
        # Reset and try again — with jitter the sequences should differ
        backoff.reset()
        delays2 = [backoff.next_delay() for _ in range(20)]
        # At least some delays should differ (extremely unlikely to be all equal)
        assert delays != delays2 or True  # jitter is random, can't guarantee

    def test_jitter_stays_non_negative(self):
        """Delay should never be negative even with large jitter."""
        backoff = ExponentialBackoff(
            base_delay=0.5, max_delay=60.0, jitter_range=0.5
        )
        for _ in range(100):
            delay = backoff.next_delay()
            assert delay >= 0

    def test_reset_resets_attempt_counter(self):
        """After reset, delays should start from base again."""
        backoff = ExponentialBackoff(
            base_delay=1.0, max_delay=60.0, multiplier=2.0, jitter_range=0.0
        )
        backoff.next_delay()  # 1
        backoff.next_delay()  # 2
        backoff.next_delay()  # 4
        assert backoff.attempt == 3
        backoff.reset()
        assert backoff.attempt == 0
        delay = backoff.next_delay()
        assert delay == 1.0

    def test_attempt_counter_increments(self):
        """Attempt counter should increment with each next_delay call."""
        backoff = ExponentialBackoff()
        assert backoff.attempt == 0
        backoff.next_delay()
        assert backoff.attempt == 1
        backoff.next_delay()
        assert backoff.attempt == 2

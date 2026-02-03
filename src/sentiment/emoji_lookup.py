"""
Emoji sentiment modifiers for financial text analysis.

Maps emojis commonly used in financial social media to sentiment adjustments.
Bullish emojis increase positive sentiment scores, bearish emojis decrease them.

Usage:
    from src.sentiment.emoji_lookup import compute_emoji_modifier, extract_emojis

    modifier = compute_emoji_modifier("NVDA 🚀🚀📈")  # Returns ~0.5
    emojis = extract_emojis("Moon 🌙 soon 🚀")  # Returns ["🌙", "🚀"]
"""

import re
from typing import Dict, List, Tuple

# Emoji sentiment weights: positive = bullish, negative = bearish
# Values are additive modifiers clamped to [-max_modifier, +max_modifier]
EMOJI_SENTIMENT: Dict[str, float] = {
    # Bullish emojis (positive weights)
    "🚀": 0.3,      # Rocket - strong bullish signal
    "📈": 0.2,      # Chart increasing
    "💎": 0.1,      # Diamond (diamond hands)
    "🙌": 0.1,      # Raising hands (often paired with diamond)
    "💰": 0.15,     # Money bag
    "🤑": 0.15,     # Money face
    "💪": 0.1,      # Strength
    "🎯": 0.05,     # Target/bullseye
    "✅": 0.05,     # Check mark
    "🏆": 0.1,      # Trophy
    "⬆️": 0.1,      # Up arrow
    "🟢": 0.1,      # Green circle
    "🌙": 0.15,     # Moon (to the moon)
    "🔥": 0.15,     # Fire (hot stock)
    "💵": 0.1,      # Dollar
    "🤝": 0.05,     # Handshake
    "👍": 0.05,     # Thumbs up
    "🎉": 0.1,      # Celebration
    "⭐": 0.05,     # Star
    "💯": 0.1,      # 100 percent
    # Bearish emojis (negative weights)
    "📉": -0.2,     # Chart decreasing
    "💩": -0.2,     # Poop
    "🤡": -0.1,     # Clown
    "⚠️": -0.1,     # Warning
    "❌": -0.2,     # X mark
    "🔻": -0.15,    # Red triangle down
    "⬇️": -0.1,     # Down arrow
    "🩸": -0.15,    # Blood (bleeding)
    "💀": -0.2,     # Skull
    "🪦": -0.25,    # Tombstone
    "🗑️": -0.2,     # Trash
    "🔴": -0.1,     # Red circle
    "😢": -0.1,     # Crying
    "😱": -0.15,    # Screaming
    "😰": -0.1,     # Anxious
    "👎": -0.05,    # Thumbs down
    "🐻": -0.2,     # Bear (bearish)
    "💔": -0.1,     # Broken heart
    "🆘": -0.15,    # SOS
    "☠️": -0.2,     # Skull and crossbones
}

# Regex pattern for extracting emojis
# Covers most common emoji ranges including compound emojis
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0001F900-\U0001F9FF"  # Supplemental symbols
    "\U0001FA00-\U0001FA6F"  # Chess, etc.
    "\U0001FA70-\U0001FAFF"  # Symbols extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002300-\U000023FF"  # Misc technical
    "]+",
    flags=re.UNICODE,
)


def extract_emojis(text: str) -> List[str]:
    """
    Extract all emojis from text.

    Args:
        text: Input text containing emojis

    Returns:
        List of emoji strings found in text (may include compound emojis)
    """
    if not text:
        return []

    matches = _EMOJI_PATTERN.findall(text)
    # Split compound emoji matches into individual emojis
    emojis = []
    for match in matches:
        # Handle variation selectors and ZWJ sequences
        emojis.extend(list(match))

    return emojis


def get_emoji_breakdown(text: str) -> Dict[str, float]:
    """
    Get sentiment breakdown for all emojis in text.

    Args:
        text: Input text containing emojis

    Returns:
        Dictionary mapping each found emoji to its sentiment weight.
        Emojis not in EMOJI_SENTIMENT are excluded.
    """
    emojis = extract_emojis(text)
    breakdown = {}

    for emoji in emojis:
        if emoji in EMOJI_SENTIMENT:
            # Count duplicates by incrementing
            if emoji in breakdown:
                breakdown[emoji] += EMOJI_SENTIMENT[emoji]
            else:
                breakdown[emoji] = EMOJI_SENTIMENT[emoji]

    return breakdown


def compute_emoji_modifier(text: str, max_modifier: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """
    Compute aggregate emoji sentiment modifier for text.

    Sums individual emoji weights and clamps to [-max_modifier, +max_modifier].
    This modifier can be used to adjust model confidence scores.

    Args:
        text: Input text containing emojis
        max_modifier: Maximum absolute adjustment value (default 0.5)

    Returns:
        Tuple of (modifier, emoji_breakdown):
        - modifier: Clamped aggregate modifier in [-max_modifier, +max_modifier]
        - emoji_breakdown: Dict mapping each emoji to its weight contribution
    """
    if max_modifier < 0:
        raise ValueError("max_modifier must be non-negative")

    breakdown = get_emoji_breakdown(text)

    if not breakdown:
        return 0.0, {}

    # Sum all emoji contributions
    raw_modifier = sum(breakdown.values())

    # Clamp to bounds
    clamped_modifier = max(-max_modifier, min(max_modifier, raw_modifier))

    return clamped_modifier, breakdown


def get_emoji_sentiment(emoji: str) -> float:
    """
    Get sentiment weight for a single emoji.

    Args:
        emoji: Single emoji character

    Returns:
        Sentiment weight (positive for bullish, negative for bearish).
        Returns 0.0 if emoji is not in the lookup table.
    """
    return EMOJI_SENTIMENT.get(emoji, 0.0)


def is_bullish_emoji(emoji: str) -> bool:
    """Check if emoji has bullish (positive) sentiment."""
    return EMOJI_SENTIMENT.get(emoji, 0.0) > 0


def is_bearish_emoji(emoji: str) -> bool:
    """Check if emoji has bearish (negative) sentiment."""
    return EMOJI_SENTIMENT.get(emoji, 0.0) < 0

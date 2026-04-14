"""
Curated list of semiconductor-focused X/Twitter accounts for browser ingestion.

These accounts are tracked by the Twitter adapter xui path when no explicit source
override is provided.
"""

# Semiconductor analysts and researchers
ANALYST_ACCOUNTS = [
    "SemiAnalysis",  # SemiAnalysis - deep semiconductor analysis
    "chinabeige",  # Dylan Patel - semiconductor supply chain
    "patrick_adventures",  # Patrick Moorhead - tech analyst
    "daniel_sumi",  # Daniel Sumi - semiconductor industry
    "TechAltar",  # TechAltar - chip industry analysis
]

# Major semiconductor company accounts
COMPANY_ACCOUNTS = [
    "nvidia",  # NVIDIA Corporation
    "AMD",  # Advanced Micro Devices
    "intel",  # Intel Corporation
    "Qualcomm",  # Qualcomm
    "Broadcom",  # Broadcom Inc
    "MicronTech",  # Micron Technology
    "Samsung_SD",  # Samsung Semiconductor
    "SKhynix",  # SK Hynix
]

# Market and trading focused accounts
MARKET_ACCOUNTS = [
    "unusual_whales",  # Unusual options activity
    "StockMKTNewz",  # Market news aggregator
    "DeItaone",  # Breaking financial news
]

# All default accounts combined
DEFAULT_USERNAMES = ANALYST_ACCOUNTS + COMPANY_ACCOUNTS + MARKET_ACCOUNTS


def get_default_usernames() -> list[str]:
    """
    Get the default list of Twitter usernames to track via xui.

    Returns:
        List of Twitter usernames (without @ prefix)
    """
    return DEFAULT_USERNAMES.copy()


def parse_usernames(usernames_str: str | None) -> list[str]:
    """
    Parse comma-separated usernames string into a list.

    Args:
        usernames_str: Comma-separated usernames (e.g., "user1,user2,user3")

    Returns:
        List of usernames, or default list if input is None/empty
    """
    if not usernames_str:
        return get_default_usernames()

    # Split by comma, strip whitespace, filter empty strings
    usernames = [u.strip().lstrip("@") for u in usernames_str.split(",")]
    return [u for u in usernames if u]

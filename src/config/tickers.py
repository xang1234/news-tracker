"""Static ticker dictionary and company name mappings for semiconductor sector.

This module provides:
1. A curated list of semiconductor tickers to track
2. Company name -> ticker mappings for fuzzy matching
3. Common abbreviations and alternate names

When security_master_enabled is True, `init_security_master()` populates
module-level caches from the database. The public functions (`get_all_tickers`,
`normalize_ticker`, `company_to_ticker`) check these caches first, falling back
to the static dicts when the feature is disabled or the DB is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.database import Database

logger = logging.getLogger(__name__)

# Module-level caches populated by init_security_master()
_cached_tickers: set[str] | None = None
_cached_company_map: dict[str, str] | None = None

# Primary semiconductor tickers to track
SEMICONDUCTOR_TICKERS: set[str] = {
    # GPU/AI Chips
    "NVDA",  # NVIDIA
    "AMD",   # Advanced Micro Devices
    "INTC",  # Intel

    # Foundries
    "TSM",   # Taiwan Semiconductor (TSMC)
    "UMC",   # United Microelectronics
    "GFS",   # GlobalFoundries

    # Memory
    "MU",    # Micron Technology
    "WDC",   # Western Digital
    "STX",   # Seagate (HDD but relevant)

    # Equipment
    "ASML",  # ASML Holding (lithography)
    "AMAT",  # Applied Materials
    "LRCX",  # Lam Research
    "KLAC",  # KLA Corporation
    "TER",   # Teradyne

    # Analog/Mixed-Signal
    "TXN",   # Texas Instruments
    "ADI",   # Analog Devices
    "MCHP",  # Microchip Technology
    "ON",    # ON Semiconductor
    "NXPI",  # NXP Semiconductors

    # Mobile/Wireless
    "QCOM",  # Qualcomm
    "AVGO",  # Broadcom
    "MRVL",  # Marvell Technology
    "SWKS",  # Skyworks Solutions
    "QRVO",  # Qorvo

    # Specialty
    "MPWR",  # Monolithic Power Systems
    "CRUS",  # Cirrus Logic
    "SLAB",  # Silicon Labs
    "WOLF",  # Wolfspeed (SiC)
    "LSCC",  # Lattice Semiconductor
    "SMTC",  # Semtech

    # AI Infrastructure (adjacent)
    "ARM",   # ARM Holdings
    "SNPS",  # Synopsys (EDA)
    "CDNS",  # Cadence Design (EDA)
}

# Company name to ticker mapping for fuzzy matching
# Keys are lowercase for case-insensitive lookup
COMPANY_TO_TICKER: dict[str, str] = {
    # NVIDIA
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "geforce": "NVDA",
    "jensen huang": "NVDA",
    "jensen": "NVDA",

    # AMD
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "lisa su": "AMD",
    "radeon": "AMD",
    "ryzen": "AMD",
    "epyc": "AMD",
    "xilinx": "AMD",  # Acquired by AMD

    # Intel
    "intel": "INTC",
    "intc": "INTC",
    "pat gelsinger": "INTC",
    "xeon": "INTC",
    "core i": "INTC",
    "arc gpu": "INTC",
    "altera": "INTC",  # Acquired by Intel

    # TSMC
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "taiwan semi": "TSM",
    "morris chang": "TSM",
    "c.c. wei": "TSM",

    # Micron
    "micron": "MU",
    "micron technology": "MU",
    "crucial": "MU",  # Micron brand

    # ASML
    "asml": "ASML",
    "euv": "ASML",  # EUV lithography strongly associated

    # Applied Materials
    "applied materials": "AMAT",
    "amat": "AMAT",

    # Lam Research
    "lam research": "LRCX",
    "lam": "LRCX",

    # KLA
    "kla": "KLAC",
    "kla corporation": "KLAC",

    # Qualcomm
    "qualcomm": "QCOM",
    "snapdragon": "QCOM",

    # Broadcom
    "broadcom": "AVGO",
    "avgo": "AVGO",
    "hock tan": "AVGO",

    # Texas Instruments
    "texas instruments": "TXN",
    "ti": "TXN",
    "txn": "TXN",

    # Analog Devices
    "analog devices": "ADI",
    "adi": "ADI",

    # NXP
    "nxp": "NXPI",
    "nxp semiconductors": "NXPI",

    # Marvell
    "marvell": "MRVL",
    "marvell technology": "MRVL",

    # ON Semiconductor
    "on semiconductor": "ON",
    "on semi": "ON",
    "onsemi": "ON",

    # Microchip
    "microchip": "MCHP",
    "microchip technology": "MCHP",

    # Skyworks
    "skyworks": "SWKS",
    "skyworks solutions": "SWKS",

    # Qorvo
    "qorvo": "QRVO",

    # ARM
    "arm": "ARM",
    "arm holdings": "ARM",
    "softbank arm": "ARM",

    # Synopsys
    "synopsys": "SNPS",

    # Cadence
    "cadence": "CDNS",
    "cadence design": "CDNS",

    # GlobalFoundries
    "globalfoundries": "GFS",
    "gf": "GFS",

    # Wolfspeed
    "wolfspeed": "WOLF",
    "cree": "WOLF",  # Former name
    "silicon carbide": "WOLF",
    "sic": "WOLF",

    # Lattice
    "lattice": "LSCC",
    "lattice semiconductor": "LSCC",

    # Memory companies
    "samsung semiconductor": "005930.KS",  # Korean ticker
    "samsung memory": "005930.KS",
    "sk hynix": "000660.KS",
    "hynix": "000660.KS",
}

# Technology keywords that indicate semiconductor relevance
SEMICONDUCTOR_KEYWORDS: set[str] = {
    # Manufacturing
    "wafer", "fab", "foundry", "lithography", "euv", "duv",
    "node", "nm process", "3nm", "5nm", "7nm", "10nm", "14nm",
    "chiplet", "packaging", "cowos", "foveros", "emib",

    # Memory
    "dram", "nand", "hbm", "gddr", "ddr5", "ddr4",
    "memory bandwidth", "memory stacking",

    # AI/ML
    "gpu", "tpu", "npu", "accelerator", "inference",
    "training chip", "ai chip", "ml chip",
    "cuda", "tensor core", "matrix engine",

    # Networking
    "networking chip", "switch asic", "nic",
    "dpu", "smartnic", "infiniband",

    # Automotive
    "automotive chip", "adas", "soc",

    # Supply chain
    "chip shortage", "semiconductor shortage",
    "chip supply", "semiconductor supply",
    "chip inventory", "semiconductor inventory",
}


def get_all_tickers() -> set[str]:
    """Get all tracked tickers.

    Returns DB-backed data when security master is initialized,
    otherwise falls back to the static SEMICONDUCTOR_TICKERS dict.
    """
    if _cached_tickers is not None:
        return _cached_tickers.copy()
    return SEMICONDUCTOR_TICKERS.copy()


def normalize_ticker(ticker: str) -> str | None:
    """Normalize a ticker symbol to standard format.

    Returns None if ticker is not in our tracked set.
    """
    ticker = ticker.upper().strip()
    if ticker.startswith("$"):
        ticker = ticker[1:]

    active = _cached_tickers if _cached_tickers is not None else SEMICONDUCTOR_TICKERS
    if ticker in active:
        return ticker
    return None


def company_to_ticker(company_name: str) -> str | None:
    """Look up ticker from company name (exact match).

    Returns None if no match found.
    """
    source = _cached_company_map if _cached_company_map is not None else COMPANY_TO_TICKER
    return source.get(company_name.lower().strip())


async def init_security_master(db: Database) -> None:
    """Load security master data from the DB into module-level caches.

    Called at application startup when security_master_enabled is True.
    Falls back silently on error so the static dicts remain available.
    """
    global _cached_tickers, _cached_company_map

    try:
        from src.security_master.service import SecurityMasterService

        svc = SecurityMasterService(db)
        await svc.ensure_seeded()
        _cached_tickers = await svc.get_all_tickers()
        _cached_company_map = await svc.get_company_map()
        logger.info(
            "Security master initialized: %d tickers, %d company aliases",
            len(_cached_tickers),
            len(_cached_company_map),
        )
    except Exception:
        logger.exception("Failed to initialize security master, using static fallback")
        _cached_tickers = None
        _cached_company_map = None


def _reset_cache() -> None:
    """Clear module-level caches (for testing)."""
    global _cached_tickers, _cached_company_map
    _cached_tickers = None
    _cached_company_map = None

"""Seed data for the semiconductor supply chain causal graph.

Populates the graph with ~100 key relationships covering:
- Foundry supply chains (TSMC, Samsung, GlobalFoundries → fabless customers)
- Equipment supply chains (ASML, AMAT, LRCX → foundries)
- Memory supply chains (SK Hynix, Samsung, Micron → AI chip makers)
- EDA/IP supply chains (Synopsys, Cadence, ARM → chip designers)
- Competition relationships across all segments
- Technology dependencies (EUV, HBM3E, CoWoS, etc.)
- Demand drivers (AI training, cloud capex, automotive)

The seed is idempotent: uses ON CONFLICT upserts so it's safe to re-run.
"""

import logging
from dataclasses import dataclass

from src.graph.schemas import CausalEdge, CausalNode, NodeType, RelationType
from src.storage.database import Database

from .causal_graph import CausalGraph

logger = logging.getLogger(__name__)

SEED_VERSION = 1

# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NodeDef:
    """Lightweight node definition for seed data."""

    node_id: str
    node_type: NodeType
    name: str
    metadata: dict


def _ticker(node_id: str, name: str, **meta: object) -> _NodeDef:
    return _NodeDef(node_id, "ticker", name, dict(meta))


def _tech(node_id: str, name: str, **meta: object) -> _NodeDef:
    return _NodeDef(node_id, "technology", name, dict(meta))


def _theme(node_id: str, name: str, **meta: object) -> _NodeDef:
    return _NodeDef(node_id, "theme", name, dict(meta))


# --- Ticker nodes ---

TICKER_NODES: list[_NodeDef] = [
    # GPU / AI Chips
    _ticker("NVDA", "NVIDIA Corporation", sector="gpu_ai"),
    _ticker("AMD", "Advanced Micro Devices", sector="gpu_ai"),
    _ticker("INTC", "Intel Corporation", sector="gpu_ai"),
    # Foundries
    _ticker("TSM", "Taiwan Semiconductor (TSMC)", sector="foundry"),
    _ticker("SAMSUNG", "Samsung Electronics", sector="foundry_memory"),
    _ticker("GFS", "GlobalFoundries", sector="foundry"),
    _ticker("UMC", "United Microelectronics", sector="foundry"),
    # Memory
    _ticker("SK_HYNIX", "SK Hynix", sector="memory"),
    _ticker("MU", "Micron Technology", sector="memory"),
    # Equipment
    _ticker("ASML", "ASML Holding", sector="equipment"),
    _ticker("AMAT", "Applied Materials", sector="equipment"),
    _ticker("LRCX", "Lam Research", sector="equipment"),
    _ticker("KLAC", "KLA Corporation", sector="equipment"),
    _ticker("TER", "Teradyne", sector="equipment"),
    # EDA / IP
    _ticker("SNPS", "Synopsys", sector="eda"),
    _ticker("CDNS", "Cadence Design Systems", sector="eda"),
    _ticker("ARM", "ARM Holdings", sector="ip"),
    # Mobile / Wireless
    _ticker("QCOM", "Qualcomm", sector="mobile"),
    _ticker("AVGO", "Broadcom", sector="networking"),
    _ticker("MRVL", "Marvell Technology", sector="networking"),
    # Analog / Mixed-Signal
    _ticker("TXN", "Texas Instruments", sector="analog"),
    _ticker("ADI", "Analog Devices", sector="analog"),
    _ticker("NXPI", "NXP Semiconductors", sector="automotive"),
    _ticker("ON", "ON Semiconductor", sector="automotive"),
    _ticker("MCHP", "Microchip Technology", sector="analog"),
    # RF
    _ticker("SWKS", "Skyworks Solutions", sector="rf"),
    _ticker("QRVO", "Qorvo", sector="rf"),
    # Power / Specialty
    _ticker("MPWR", "Monolithic Power Systems", sector="power"),
    _ticker("WOLF", "Wolfspeed", sector="sic"),
    _ticker("LSCC", "Lattice Semiconductor", sector="fpga"),
]

# --- Technology nodes ---

TECHNOLOGY_NODES: list[_NodeDef] = [
    _tech("EUV", "Extreme Ultraviolet Lithography", category="manufacturing"),
    _tech("HBM3E", "High Bandwidth Memory 3E", category="memory"),
    _tech("CoWoS", "Chip-on-Wafer-on-Substrate", category="packaging"),
    _tech("FOVEROS", "Foveros 3D Packaging", category="packaging"),
    _tech("GAA", "Gate-All-Around Transistors", category="manufacturing"),
    _tech("CHIPLET", "Chiplet Architecture", category="packaging"),
    _tech("DDR5", "DDR5 Memory Standard", category="memory"),
    _tech("SiC", "Silicon Carbide", category="material"),
    _tech("GaN", "Gallium Nitride", category="material"),
    _tech("InFO", "Integrated Fan-Out Packaging", category="packaging"),
    _tech("CUDA", "CUDA Parallel Computing", category="software"),
    _tech("3NM", "3nm Process Node", category="manufacturing"),
    _tech("5NM", "5nm Process Node", category="manufacturing"),
]

# --- Theme nodes ---

THEME_NODES: list[_NodeDef] = [
    _theme("theme_ai_training", "AI Training Demand"),
    _theme("theme_ai_inference", "AI Inference Demand"),
    _theme("theme_cloud_capex", "Cloud Capital Expenditure"),
    _theme("theme_hbm_demand", "High Bandwidth Memory Demand"),
    _theme("theme_ai_accelerators", "AI Accelerator Competition"),
    _theme("theme_automotive_chips", "Automotive Semiconductor Demand"),
    _theme("theme_advanced_packaging", "Advanced Packaging Technology"),
    _theme("theme_foundry_competition", "Foundry Competition"),
]

ALL_NODES: list[_NodeDef] = TICKER_NODES + TECHNOLOGY_NODES + THEME_NODES

# ---------------------------------------------------------------------------
# Edge definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EdgeDef:
    """Lightweight edge definition for seed data."""

    source: str
    target: str
    relation: RelationType
    confidence: float


def _edge(
    source: str,
    target: str,
    relation: RelationType,
    confidence: float = 0.9,
) -> _EdgeDef:
    return _EdgeDef(source, target, relation, confidence)


# --- Foundry supply chain ---
# TSM manufactures chips for major fabless companies

FOUNDRY_SUPPLY_EDGES: list[_EdgeDef] = [
    # TSMC → customers
    _edge("TSM", "NVDA", "supplies_to", 0.95),
    _edge("TSM", "AMD", "supplies_to", 0.95),
    _edge("TSM", "QCOM", "supplies_to", 0.95),
    _edge("TSM", "AVGO", "supplies_to", 0.90),
    _edge("TSM", "MRVL", "supplies_to", 0.85),
    _edge("TSM", "ARM", "supplies_to", 0.80),
    _edge("TSM", "INTC", "supplies_to", 0.70),  # Intel outsources some
    # Samsung Foundry → customers
    _edge("SAMSUNG", "QCOM", "supplies_to", 0.85),
    # GlobalFoundries → customers
    _edge("GFS", "AMD", "supplies_to", 0.75),  # Legacy nodes
    _edge("GFS", "QCOM", "supplies_to", 0.70),
    _edge("GFS", "NXPI", "supplies_to", 0.70),
    # UMC → customers (mature nodes)
    _edge("UMC", "MCHP", "supplies_to", 0.75),
    _edge("UMC", "TXN", "supplies_to", 0.70),
]

# --- Equipment supply chain ---
# Equipment makers → foundries

EQUIPMENT_SUPPLY_EDGES: list[_EdgeDef] = [
    # ASML (lithography) → foundries
    _edge("ASML", "TSM", "supplies_to", 0.95),
    _edge("ASML", "SAMSUNG", "supplies_to", 0.90),
    _edge("ASML", "INTC", "supplies_to", 0.90),
    # Applied Materials (deposition/etch) → foundries + memory
    _edge("AMAT", "TSM", "supplies_to", 0.90),
    _edge("AMAT", "SAMSUNG", "supplies_to", 0.85),
    _edge("AMAT", "INTC", "supplies_to", 0.85),
    _edge("AMAT", "SK_HYNIX", "supplies_to", 0.80),
    _edge("AMAT", "MU", "supplies_to", 0.80),
    # Lam Research (etch/deposition) → foundries + memory
    _edge("LRCX", "TSM", "supplies_to", 0.90),
    _edge("LRCX", "SAMSUNG", "supplies_to", 0.85),
    _edge("LRCX", "INTC", "supplies_to", 0.80),
    _edge("LRCX", "SK_HYNIX", "supplies_to", 0.80),
    _edge("LRCX", "MU", "supplies_to", 0.80),
    # KLA (inspection) → foundries
    _edge("KLAC", "TSM", "supplies_to", 0.85),
    _edge("KLAC", "SAMSUNG", "supplies_to", 0.80),
    _edge("KLAC", "INTC", "supplies_to", 0.80),
    # Teradyne (testing) → foundries + memory
    _edge("TER", "TSM", "supplies_to", 0.80),
    _edge("TER", "SAMSUNG", "supplies_to", 0.75),
    _edge("TER", "SK_HYNIX", "supplies_to", 0.70),
]

# --- Memory supply chain ---
# Memory makers → AI chip consumers (HBM for GPUs)

MEMORY_SUPPLY_EDGES: list[_EdgeDef] = [
    _edge("SK_HYNIX", "NVDA", "supplies_to", 0.95),   # Primary HBM supplier
    _edge("SK_HYNIX", "AMD", "supplies_to", 0.80),
    _edge("SAMSUNG", "NVDA", "supplies_to", 0.85),     # HBM + GDDR
    _edge("SAMSUNG", "AMD", "supplies_to", 0.75),
    _edge("MU", "NVDA", "supplies_to", 0.75),           # HBM3E qualification
    _edge("MU", "AMD", "supplies_to", 0.70),
    _edge("MU", "INTC", "supplies_to", 0.70),
]

# --- EDA / IP supply chain ---
# Design tools and IP blocks → chip designers

EDA_SUPPLY_EDGES: list[_EdgeDef] = [
    # Synopsys → chip designers
    _edge("SNPS", "NVDA", "supplies_to", 0.90),
    _edge("SNPS", "AMD", "supplies_to", 0.90),
    _edge("SNPS", "QCOM", "supplies_to", 0.85),
    _edge("SNPS", "INTC", "supplies_to", 0.85),
    _edge("SNPS", "AVGO", "supplies_to", 0.80),
    # Cadence → chip designers
    _edge("CDNS", "NVDA", "supplies_to", 0.90),
    _edge("CDNS", "AMD", "supplies_to", 0.85),
    _edge("CDNS", "QCOM", "supplies_to", 0.85),
    _edge("CDNS", "INTC", "supplies_to", 0.85),
    # ARM → licensees
    _edge("ARM", "QCOM", "supplies_to", 0.95),  # Snapdragon uses ARM cores
    _edge("ARM", "NVDA", "supplies_to", 0.85),   # Grace CPU
    _edge("ARM", "MRVL", "supplies_to", 0.80),
    _edge("ARM", "AVGO", "supplies_to", 0.75),
]

# --- Competition ---
# Bidirectional competition edges (A competes_with B)

COMPETITION_EDGES: list[_EdgeDef] = [
    # GPU / AI
    _edge("NVDA", "AMD", "competes_with", 0.95),
    _edge("AMD", "NVDA", "competes_with", 0.95),
    _edge("NVDA", "INTC", "competes_with", 0.80),
    _edge("INTC", "NVDA", "competes_with", 0.80),
    _edge("AMD", "INTC", "competes_with", 0.95),
    _edge("INTC", "AMD", "competes_with", 0.95),
    # Foundry
    _edge("TSM", "SAMSUNG", "competes_with", 0.90),
    _edge("SAMSUNG", "TSM", "competes_with", 0.90),
    _edge("TSM", "INTC", "competes_with", 0.70),
    _edge("INTC", "TSM", "competes_with", 0.70),
    # Memory
    _edge("SK_HYNIX", "SAMSUNG", "competes_with", 0.90),
    _edge("SAMSUNG", "SK_HYNIX", "competes_with", 0.90),
    _edge("SK_HYNIX", "MU", "competes_with", 0.85),
    _edge("MU", "SK_HYNIX", "competes_with", 0.85),
    _edge("SAMSUNG", "MU", "competes_with", 0.85),
    _edge("MU", "SAMSUNG", "competes_with", 0.85),
    # EDA
    _edge("SNPS", "CDNS", "competes_with", 0.90),
    _edge("CDNS", "SNPS", "competes_with", 0.90),
    # Networking
    _edge("AVGO", "MRVL", "competes_with", 0.80),
    _edge("MRVL", "AVGO", "competes_with", 0.80),
    # Analog / Automotive
    _edge("TXN", "ADI", "competes_with", 0.85),
    _edge("ADI", "TXN", "competes_with", 0.85),
    _edge("ON", "NXPI", "competes_with", 0.80),
    _edge("NXPI", "ON", "competes_with", 0.80),
    # RF
    _edge("SWKS", "QRVO", "competes_with", 0.85),
    _edge("QRVO", "SWKS", "competes_with", 0.85),
    # Equipment
    _edge("AMAT", "LRCX", "competes_with", 0.80),
    _edge("LRCX", "AMAT", "competes_with", 0.80),
]

# --- Technology dependencies ---
# Foundries and chip makers depend on specific technologies

TECHNOLOGY_EDGES: list[_EdgeDef] = [
    # EUV lithography
    _edge("TSM", "EUV", "depends_on", 0.95),
    _edge("SAMSUNG", "EUV", "depends_on", 0.90),
    _edge("INTC", "EUV", "depends_on", 0.90),
    _edge("ASML", "EUV", "supplies_to", 0.95),  # ASML is EUV's sole supplier
    # Advanced packaging
    _edge("TSM", "CoWoS", "supplies_to", 0.95),  # TSMC manufactures CoWoS
    _edge("NVDA", "CoWoS", "depends_on", 0.95),
    _edge("AMD", "CoWoS", "depends_on", 0.80),
    _edge("TSM", "InFO", "supplies_to", 0.90),
    _edge("INTC", "FOVEROS", "supplies_to", 0.90),
    # HBM3E
    _edge("SK_HYNIX", "HBM3E", "supplies_to", 0.95),
    _edge("SAMSUNG", "HBM3E", "supplies_to", 0.85),
    _edge("MU", "HBM3E", "supplies_to", 0.75),
    _edge("NVDA", "HBM3E", "depends_on", 0.95),
    _edge("AMD", "HBM3E", "depends_on", 0.80),
    # Process nodes
    _edge("TSM", "3NM", "supplies_to", 0.95),
    _edge("SAMSUNG", "3NM", "supplies_to", 0.80),
    _edge("TSM", "5NM", "supplies_to", 0.95),
    _edge("SAMSUNG", "5NM", "supplies_to", 0.85),
    # Chiplet architecture
    _edge("AMD", "CHIPLET", "depends_on", 0.90),
    _edge("INTC", "CHIPLET", "depends_on", 0.85),
    # Gate-all-around
    _edge("TSM", "GAA", "depends_on", 0.85),
    _edge("SAMSUNG", "GAA", "depends_on", 0.85),
    _edge("INTC", "GAA", "depends_on", 0.80),
    # DDR5
    _edge("SK_HYNIX", "DDR5", "supplies_to", 0.90),
    _edge("SAMSUNG", "DDR5", "supplies_to", 0.90),
    _edge("MU", "DDR5", "supplies_to", 0.90),
    # CUDA ecosystem
    _edge("NVDA", "CUDA", "supplies_to", 0.95),
    # Wide-bandgap materials
    _edge("WOLF", "SiC", "supplies_to", 0.95),
    _edge("ON", "SiC", "supplies_to", 0.80),
    _edge("WOLF", "GaN", "supplies_to", 0.70),
]

# --- Demand drivers ---
# Themes that drive demand for specific companies and technologies

DEMAND_DRIVER_EDGES: list[_EdgeDef] = [
    # AI training → GPU makers and memory
    _edge("theme_ai_training", "NVDA", "drives", 0.95),
    _edge("theme_ai_training", "AMD", "drives", 0.80),
    _edge("theme_ai_training", "HBM3E", "drives", 0.90),
    _edge("theme_ai_training", "CoWoS", "drives", 0.85),
    # AI inference → broader set of chip makers
    _edge("theme_ai_inference", "NVDA", "drives", 0.90),
    _edge("theme_ai_inference", "AMD", "drives", 0.80),
    _edge("theme_ai_inference", "INTC", "drives", 0.70),
    _edge("theme_ai_inference", "QCOM", "drives", 0.70),
    _edge("theme_ai_inference", "AVGO", "drives", 0.75),
    _edge("theme_ai_inference", "MRVL", "drives", 0.70),
    # Cloud capex → AI training (cascading demand)
    _edge("theme_cloud_capex", "theme_ai_training", "drives", 0.90),
    _edge("theme_cloud_capex", "theme_ai_inference", "drives", 0.85),
    _edge("theme_cloud_capex", "AVGO", "drives", 0.80),
    _edge("theme_cloud_capex", "MRVL", "drives", 0.75),
    # HBM demand → memory suppliers
    _edge("theme_hbm_demand", "SK_HYNIX", "drives", 0.95),
    _edge("theme_hbm_demand", "SAMSUNG", "drives", 0.85),
    _edge("theme_hbm_demand", "MU", "drives", 0.80),
    _edge("theme_hbm_demand", "HBM3E", "drives", 0.95),
    # AI accelerator competition
    _edge("theme_ai_accelerators", "NVDA", "drives", 0.95),
    _edge("theme_ai_accelerators", "AMD", "drives", 0.85),
    _edge("theme_ai_accelerators", "INTC", "drives", 0.70),
    _edge("theme_ai_accelerators", "AVGO", "drives", 0.75),
    # Automotive demand
    _edge("theme_automotive_chips", "ON", "drives", 0.90),
    _edge("theme_automotive_chips", "NXPI", "drives", 0.90),
    _edge("theme_automotive_chips", "TXN", "drives", 0.80),
    _edge("theme_automotive_chips", "MCHP", "drives", 0.75),
    _edge("theme_automotive_chips", "SiC", "drives", 0.85),
    _edge("theme_automotive_chips", "WOLF", "drives", 0.80),
    # Advanced packaging theme
    _edge("theme_advanced_packaging", "CoWoS", "drives", 0.95),
    _edge("theme_advanced_packaging", "FOVEROS", "drives", 0.85),
    _edge("theme_advanced_packaging", "InFO", "drives", 0.80),
    _edge("theme_advanced_packaging", "CHIPLET", "drives", 0.85),
    _edge("theme_advanced_packaging", "TSM", "drives", 0.90),
    # Foundry competition theme
    _edge("theme_foundry_competition", "TSM", "drives", 0.90),
    _edge("theme_foundry_competition", "SAMSUNG", "drives", 0.85),
    _edge("theme_foundry_competition", "INTC", "drives", 0.80),
    _edge("theme_foundry_competition", "GFS", "drives", 0.70),
    _edge("theme_foundry_competition", "EUV", "drives", 0.85),
    _edge("theme_foundry_competition", "GAA", "drives", 0.80),
]

ALL_EDGES: list[_EdgeDef] = (
    FOUNDRY_SUPPLY_EDGES
    + EQUIPMENT_SUPPLY_EDGES
    + MEMORY_SUPPLY_EDGES
    + EDA_SUPPLY_EDGES
    + COMPETITION_EDGES
    + TECHNOLOGY_EDGES
    + DEMAND_DRIVER_EDGES
)


# ---------------------------------------------------------------------------
# Seed function
# ---------------------------------------------------------------------------


async def seed_graph(database: Database) -> dict[str, int]:
    """Populate the causal graph with semiconductor supply chain relationships.

    Idempotent: uses ON CONFLICT upserts, safe to re-run.

    Args:
        database: Connected Database instance.

    Returns:
        Summary dict with node_count and edge_count.
    """
    graph = CausalGraph(database)

    # Upsert all nodes
    node_count = 0
    for node_def in ALL_NODES:
        await graph.ensure_node(
            node_id=node_def.node_id,
            node_type=node_def.node_type,
            name=node_def.name,
            metadata=node_def.metadata,
        )
        node_count += 1

    logger.info("Seeded %d nodes", node_count)

    # Add all edges
    edge_count = 0
    for edge_def in ALL_EDGES:
        await graph.add_edge(
            source=edge_def.source,
            target=edge_def.target,
            relation=edge_def.relation,
            confidence=edge_def.confidence,
        )
        edge_count += 1

    logger.info("Seeded %d edges", edge_count)

    return {
        "seed_version": SEED_VERSION,
        "node_count": node_count,
        "edge_count": edge_count,
    }

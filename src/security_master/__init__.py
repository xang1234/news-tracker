"""Security master: database-backed ticker/company registry with fuzzy lookup.

Extended with a canonical concept registry, alias model, and
issuer/security crosswalks for the intelligence layer.
"""

from src.security_master.concept_repository import ConceptRepository
from src.security_master.concept_schemas import (
    VALID_ALIAS_TYPES,
    VALID_CONCEPT_TYPES,
    VALID_ISSUER_SECURITY_RELATIONSHIPS,
    VALID_RELATIONSHIP_TYPES,
    VALID_THEME_LINK_TYPES,
    Concept,
    ConceptAlias,
    ConceptRelationship,
    ConceptThemeLink,
    IssuerSecurityLink,
    make_concept_id,
)
from src.security_master.config import SecurityMasterConfig
from src.security_master.nasdaq_trader import (
    NASDAQ_TRADER_EXTERNAL_KEY,
    NasdaqTraderReconciliationResult,
    NasdaqTraderSymbolDirectory,
    NasdaqTraderSymbolRecord,
    parse_nasdaq_trader_symbol_directories,
)
from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security, SecurityIdentifierLineage, normalize_sec_cik
from src.security_master.service import SecurityMasterService

__all__ = [
    # Concept registry
    "VALID_ALIAS_TYPES",
    "VALID_CONCEPT_TYPES",
    "VALID_ISSUER_SECURITY_RELATIONSHIPS",
    "VALID_RELATIONSHIP_TYPES",
    "VALID_THEME_LINK_TYPES",
    "Concept",
    "ConceptAlias",
    "ConceptRelationship",
    "ConceptRepository",
    "ConceptThemeLink",
    "IssuerSecurityLink",
    "make_concept_id",
    # Security master
    "NASDAQ_TRADER_EXTERNAL_KEY",
    "NasdaqTraderReconciliationResult",
    "NasdaqTraderSymbolDirectory",
    "NasdaqTraderSymbolRecord",
    "Security",
    "SecurityIdentifierLineage",
    "SecurityMasterConfig",
    "SecurityMasterRepository",
    "SecurityMasterService",
    "normalize_sec_cik",
    "parse_nasdaq_trader_symbol_directories",
]

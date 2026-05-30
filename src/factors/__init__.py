"""Macro and supply-chain factor registry and observations."""

from src.factors.ingestion import FactorIngestionResult, FactorIngestionService
from src.factors.macro_catalog import CATALOG_VERSION, get_curated_macro_factor_series
from src.factors.providers import (
    BeaFactorProvider,
    BlsFactorProvider,
    FederalReserveCsvFactorProvider,
    FredFactorProvider,
    MacroProviderCredentials,
    MacroProviderError,
    MissingProviderCredentialError,
    ProviderResponseError,
    TreasuryFiscalDataProvider,
)
from src.factors.refresh import (
    FactorRefreshSummary,
    curated_factor_series,
    provider_names,
    refresh_curated_factor_series,
)
from src.factors.regimes import (
    FactorRegimeContext,
    FactorRegimeService,
    classify_factor_regime,
)
from src.factors.repository import FactorRepository
from src.factors.schemas import (
    VALID_FACTOR_CADENCES,
    FactorObservation,
    FactorSeries,
    validate_observation_for_series,
)
from src.factors.supply_chain_catalog import (
    SUPPLY_CHAIN_CATALOG_VERSION,
    get_curated_supply_chain_factor_series,
)
from src.factors.supply_chain_providers import (
    CensusTradeFactorProvider,
    EiaFactorProvider,
    SupplyChainProviderCredentials,
)

__all__ = [
    "CATALOG_VERSION",
    "SUPPLY_CHAIN_CATALOG_VERSION",
    "VALID_FACTOR_CADENCES",
    "BeaFactorProvider",
    "BlsFactorProvider",
    "CensusTradeFactorProvider",
    "EiaFactorProvider",
    "FactorIngestionResult",
    "FactorIngestionService",
    "FactorObservation",
    "FactorRegimeContext",
    "FactorRegimeService",
    "FactorRepository",
    "FactorRefreshSummary",
    "FactorSeries",
    "FederalReserveCsvFactorProvider",
    "FredFactorProvider",
    "MacroProviderCredentials",
    "MacroProviderError",
    "MissingProviderCredentialError",
    "ProviderResponseError",
    "SupplyChainProviderCredentials",
    "TreasuryFiscalDataProvider",
    "classify_factor_regime",
    "curated_factor_series",
    "get_curated_macro_factor_series",
    "get_curated_supply_chain_factor_series",
    "provider_names",
    "refresh_curated_factor_series",
    "validate_observation_for_series",
]

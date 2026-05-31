"""Parse SEC Company Facts payloads into typed fact observations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from src.filing.sec_delta_models import SECFactObservation
from src.filing.sec_structured import SECStructuredPayloadRecord
from src.security_master.schemas import normalize_sec_cik


def extract_companyfacts_observations(
    companyfacts: SECStructuredPayloadRecord,
) -> list[SECFactObservation]:
    """Extract typed observations from a SEC Company Facts cache record."""
    cik = normalize_sec_cik(companyfacts.cik)
    if cik is None:
        raise ValueError("companyfacts record must have a valid CIK")

    observations: list[SECFactObservation] = []
    facts = companyfacts.payload.get("facts", {})
    if not isinstance(facts, dict):
        return observations

    for taxonomy, taxonomy_payload in facts.items():
        if not isinstance(taxonomy_payload, dict):
            continue
        for fact_name, fact_payload in taxonomy_payload.items():
            if not isinstance(fact_payload, dict):
                continue
            units = fact_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            observations.extend(
                _extract_unit_observations(
                    cik=companyfacts.cik,
                    taxonomy=str(taxonomy),
                    fact_name=str(fact_name),
                    units=units,
                    source=companyfacts,
                )
            )
    return observations


def _extract_unit_observations(
    *,
    cik: str,
    taxonomy: str,
    fact_name: str,
    units: dict[str, Any],
    source: SECStructuredPayloadRecord,
) -> list[SECFactObservation]:
    observations: list[SECFactObservation] = []
    for unit, unit_values in units.items():
        if not isinstance(unit_values, list):
            continue
        for raw_observation in unit_values:
            observation = _parse_observation(
                cik=cik,
                taxonomy=taxonomy,
                fact_name=fact_name,
                unit=str(unit),
                raw_observation=raw_observation,
                source=source,
            )
            if observation is not None:
                observations.append(observation)
    return observations


def _parse_observation(
    *,
    cik: str,
    taxonomy: str,
    fact_name: str,
    unit: str,
    raw_observation: Any,
    source: SECStructuredPayloadRecord,
) -> SECFactObservation | None:
    if not isinstance(raw_observation, dict):
        return None
    try:
        value = Decimal(str(raw_observation["val"]))
        accession_number = str(raw_observation["accn"])
        filed_date = date.fromisoformat(str(raw_observation["filed"]))
        period_end = date.fromisoformat(str(raw_observation["end"]))
    except (KeyError, ValueError, InvalidOperation):
        return None

    period_start = _parse_optional_date(raw_observation.get("start"))
    return SECFactObservation(
        cik=cik,
        taxonomy=taxonomy,
        fact_name=fact_name,
        unit=unit,
        accession_number=accession_number,
        form=str(raw_observation.get("form") or ""),
        filed_date=filed_date,
        period_start=period_start,
        period_end=period_end,
        value=value,
        fy=_parse_optional_int(raw_observation.get("fy")),
        fp=str(raw_observation.get("fp")) if raw_observation.get("fp") is not None else None,
        frame=str(raw_observation.get("frame"))
        if raw_observation.get("frame") is not None
        else None,
        fetched_at=source.fetched_at,
        source_payload_hash=source.payload_hash,
        source_url=source.source_url,
    )


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

"""Pattern-based event extraction for semiconductor financial text.

Uses compiled regex patterns with named capture groups to extract
structured events (SVO triplets) from document text. Each event type
has multiple patterns tuned for financial news language.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.normalizer import TimeNormalizer
from src.event_extraction.schemas import EventRecord, VALID_EVENT_TYPES

if TYPE_CHECKING:
    from src.ingestion.schemas import NormalizedDocument

logger = logging.getLogger(__name__)


# Reusable pattern fragments
_ACTOR = r"(?P<actor>[A-Z][A-Za-z&\s\.]{1,40}?)"
_QUANTITY = r"(?P<quantity>\$?\d[\d,\.]*\s*(?:billion|million|B|M|%|nm|units?)?)"
_TIME = r"(?:(?:in|by|during|for|starting)\s+)?(?P<time_ref>Q[1-4]\s*\d{4}|H[12]\s*\d{4}|next\s+(?:quarter|year|month)|(?:end\s+of\s+(?:the\s+)?year|year[\s-]?end)|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[\s,]*\d{4}?)"


def _build_patterns() -> dict[str, list[re.Pattern[str]]]:
    """
    Build and compile regex patterns for each event type.

    Returns:
        Dict mapping event_type to list of compiled patterns.
    """
    patterns: dict[str, list[str]] = {
        "capacity_expansion": [
            # "TSMC is expanding fab capacity"
            rf"{_ACTOR}\s+(?:is\s+)?(?P<action>expand(?:s|ing|ed)?)\s+(?P<object>(?:fab|fabrication|production|manufacturing|wafer|chip|capacity)[A-Za-z\s]{{0,30}}?)(?:\s+{_TIME})?",
            # "new fab / new facility / new plant"
            rf"{_ACTOR}\s+(?P<action>(?:plans?|announced?|builds?|is\s+building|will\s+build|to\s+build|open(?:s|ing|ed)?|construct(?:s|ing|ed)?))\s+(?:a\s+)?(?P<object>new\s+(?:fab|fabrication\s+facility|plant|factory|facility|foundry)[A-Za-z\s]{{0,30}}?)(?:\s+{_TIME})?",
            # "investing $X in capacity / production"
            rf"{_ACTOR}\s+(?P<action>invest(?:s|ing|ed)?)\s+{_QUANTITY}\s+(?:in\s+)?(?P<object>[A-Za-z\s]{{2,40}}?)(?:\s+{_TIME})?",
            # "capacity increase / ramp-up / production ramp"
            rf"{_ACTOR}\s+(?P<action>(?:increas|ramp|boost|scale|doubl|tripl)(?:es?|s|ing|ed)?(?:\s+up)?)\s+(?P<object>(?:production|manufacturing|output|capacity|wafer\s+starts?)[A-Za-z\s]{{0,30}}?)(?:\s+{_TIME})?",
            # "$X billion investment in ..."
            rf"{_ACTOR}\s+(?P<action>announc(?:es?|ed|ing))\s+(?:a\s+)?{_QUANTITY}\s+(?P<object>(?:investment|expansion|upgrade)[A-Za-z\s]{{0,30}}?)(?:\s+{_TIME})?",
            # Passive: "capacity is being expanded"
            r"(?P<object>[A-Za-z\s]{2,40}?)\s+capacity\s+(?:is\s+)?(?:being\s+)?(?P<action>expand(?:ed|ing)|increas(?:ed|ing)|ramp(?:ed|ing)\s+up)",
            # "plans to add X capacity"
            rf"{_ACTOR}\s+(?P<action>plans?\s+to\s+add)\s+(?:{_QUANTITY}\s+)?(?P<object>[A-Za-z\s]{{2,40}}?\s*capacity)",
            # "adding production lines"
            rf"{_ACTOR}\s+(?:is\s+)?(?P<action>add(?:s|ing|ed)?)\s+(?P<object>(?:new\s+)?(?:production|manufacturing)\s+lines?[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "capacity will reach / grow to X"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>capacity)\s+(?P<action>(?:will\s+)?(?:reach|grow\s+to|increase\s+to))\s+{_QUANTITY}(?:\s+{_TIME})?",
            # "wafer starts to increase"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>wafer\s+starts?)\s+(?:(?:expected|projected|set)\s+)?(?P<action>to\s+(?:increase|rise|grow|double|triple))(?:\s+{_TIME})?",
        ],
        "capacity_constraint": [
            # "supply shortage / supply constraints"
            rf"(?P<object>[A-Za-z\s]{{2,40}}?)\s+(?P<action>(?:supply\s+)?(?:shortage|constraint|bottleneck|deficit|undersupply|tight(?:ness|ening)?))(?:\s+{_TIME})?",
            # "TSMC faces supply constraints"
            rf"{_ACTOR}\s+(?P<action>fac(?:es?|ing)|report(?:s|ing|ed)?|experienc(?:es?|ing)|encounter(?:s|ing|ed)?)\s+(?P<object>(?:supply\s+)?(?:shortage|constraint|bottleneck|production\s+issue|capacity\s+limit)[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "lead times extended / increasing"
            r"(?P<object>lead\s+times?)\s+(?:are\s+)?(?:being\s+)?(?P<action>extend(?:ed|ing)|increas(?:ed|ing)|stretch(?:ed|ing)|lengthen(?:ed|ing))(?:\s+to\s+(?P<quantity>\d[\d,\.]*\s*(?:weeks?|months?)))?",
            # "production halted / suspended / cut"
            rf"{_ACTOR}\s+(?P<action>halt(?:s|ed|ing)?|suspend(?:s|ed|ing)?|cut(?:s|ting)?|reduc(?:es?|ed|ing)?)\s+(?P<object>(?:production|manufacturing|output|wafer\s+starts?)[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "allocation / rationing"
            rf"{_ACTOR}\s+(?P<action>(?:is\s+)?(?:allocat|ration)(?:es?|ing|ed)?)\s+(?P<object>[A-Za-z\s]{{2,40}}?)(?:\s+{_TIME})?",
            # "yield issues / yield problems"
            rf"{_ACTOR}\s+(?:(?:is\s+)?(?P<action>experienc(?:es?|ing)|report(?:s|ing|ed)?))\s+(?P<object>(?:low\s+)?yield\s+(?:issue|problem|challenge)[A-Za-z\s]{{0,15}}?)",
            # "unable to meet demand"
            rf"{_ACTOR}\s+(?:is\s+)?(?P<action>unable\s+to\s+meet)\s+(?P<object>(?:customer\s+)?demand[A-Za-z\s]{{0,15}}?)",
            # "demand exceeds supply"
            r"(?P<object>demand)\s+(?P<action>exceed(?:s|ed|ing)?|outstrip(?:s|ped|ping)?|outpac(?:es?|ed|ing)?)\s+(?:available\s+)?supply",
            # "capacity fully booked / sold out"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>capacity)\s+(?:is\s+)?(?P<action>fully\s+(?:booked|allocated|committed|sold\s+out))",
            # "limited availability"
            r"(?P<action>limited)\s+(?P<object>(?:availability|supply)\s+(?:of\s+)?[A-Za-z\s]{2,30}?)",
        ],
        "product_launch": [
            # "NVIDIA launched / launches the H200"
            rf"{_ACTOR}\s+(?P<action>launch(?:es?|ed|ing)?|introduc(?:es?|ed|ing)?|unveil(?:s|ed|ing)?|releas(?:es?|ed|ing)?|debut(?:s|ed|ing)?|roll(?:s|ed|ing)?\s+out)\s+(?:the\s+)?(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "new chip / new processor / new product"
            rf"{_ACTOR}\s+(?P<action>announc(?:es?|ed|ing))\s+(?:the\s+)?(?:new\s+)?(?P<object>(?:chip|processor|GPU|CPU|accelerator|SoC|ASIC|memory|module|platform|device|product|card|kit|server|system|node)[A-Za-z\s\-]{{0,30}}?)(?:\s+{_TIME})?",
            # "begins mass production of"
            rf"{_ACTOR}\s+(?P<action>(?:begins?|started?|commenc(?:es?|ed)?)\s+(?:mass\s+)?production\s+of)\s+(?:the\s+)?(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "sampling / taping out"
            rf"{_ACTOR}\s+(?:is\s+)?(?P<action>(?:sampl|tap)(?:es?|ing|ed)?\s*(?:out)?)\s+(?:the\s+)?(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "enters production / enters mass production"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?P<action>enter(?:s|ed|ing)?\s+(?:mass\s+)?(?:production|manufacturing|volume))(?:\s+{_TIME})?",
            # "ships first / first shipment"
            rf"{_ACTOR}\s+(?P<action>ship(?:s|ped|ping)?\s+(?:the\s+)?first)\s+(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "available now / now available"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?:is\s+)?(?P<action>(?:now\s+)?available|(?:ready\s+)?for\s+order)(?:\s+{_TIME})?",
            # "next-generation / next-gen X"
            rf"{_ACTOR}\s+(?P<action>reveal(?:s|ed|ing)?|preview(?:s|ed|ing)?)\s+(?P<object>(?:next[\s-]gen(?:eration)?\s+)?[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "starts volume shipments of"
            rf"{_ACTOR}\s+(?P<action>starts?\s+volume\s+shipments?\s+of)\s+(?:the\s+)?(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+{_TIME})?",
            # "brings X to market"
            rf"{_ACTOR}\s+(?P<action>brings?|brought)\s+(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+to\s+market(?:\s+{_TIME})?",
        ],
        "product_delay": [
            # "NVIDIA delayed the H200 / delays launch"
            rf"{_ACTOR}\s+(?P<action>delay(?:s|ed|ing)?|postpon(?:es?|ed|ing)?|push(?:es|ed|ing)?\s+back)\s+(?:the\s+)?(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)(?:\s+(?:to|until)\s+{_TIME})?",
            # "X is delayed / has been delayed"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?:is|has\s+been|was)\s+(?P<action>delay(?:ed)?|postpon(?:ed)?|push(?:ed)?\s+back)(?:\s+(?:to|until)\s+{_TIME})?",
            # "pushed to Q4 / moved to next year"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?P<action>(?:pushed|moved|shifted|rescheduled)\s+(?:to|until))\s+{_TIME}",
            # "launch / release pushed back"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?:launch|release|shipment|rollout)\s+(?P<action>delay(?:ed)?|postpon(?:ed)?|push(?:ed)?\s+back)(?:\s+(?:to|until)\s+{_TIME})?",
            # "production setback / production issues"
            rf"{_ACTOR}\s+(?P<action>(?:hit\s+(?:a|by)\s+)?(?:experienc|encounter|fac)(?:es?|ed|ing)?)\s+(?P<object>(?:production|manufacturing)\s+(?:setback|delay|issue|problem|snag)[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "behind schedule"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?:is|are|runs?|running)\s+(?P<action>behind\s+schedule)(?:\s+by\s+(?P<quantity>\d[\d,\.]*\s*(?:weeks?|months?|quarters?)))?",
            # "slip / slipped / slipping to"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?P<action>slip(?:s|ped|ping)?)\s+(?:to|until)\s+{_TIME}",
            # "supply date pushed"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?:supply|delivery|ship)\s+(?:date|timeline)\s+(?P<action>push(?:ed)?\s+(?:back|out)|delay(?:ed)?)",
            # "timeline extended"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?P<action>timeline\s+(?:extended|pushed|slipped))(?:\s+(?:to|by)\s+{_TIME})?",
            # "won't ship until"
            rf"(?P<object>[A-Za-z][\w\s\-]{{1,40}}?)\s+(?P<action>(?:won't|will\s+not)\s+(?:ship|launch|release))\s+(?:until|before)\s+{_TIME}",
        ],
        "price_change": [
            # "TSMC raised / raises prices"
            rf"{_ACTOR}\s+(?P<action>rais(?:es?|ed|ing)?|hik(?:es?|ed|ing)?|increas(?:es?|ed|ing)?)\s+(?P<object>(?:wafer\s+)?prices?[A-Za-z\s]{{0,20}}?)(?:\s+by\s+{_QUANTITY})?(?:\s+{_TIME})?",
            # "prices increased / rose / surged"
            rf"(?P<object>[A-Za-z][\w\s]{{1,40}}?)\s+(?:prices?|pricing|cost|ASP)\s+(?P<action>(?:increas|rose|surg|jump|climb|soar)(?:es?|ed|ing|s)?)(?:\s+(?:by\s+)?{_QUANTITY})?(?:\s+{_TIME})?",
            # "price cuts / lowered prices"
            rf"{_ACTOR}\s+(?P<action>cut(?:s|ting)?|lower(?:s|ed|ing)?|reduc(?:es?|ed|ing)?|slash(?:es?|ed|ing)?|discount(?:s|ed|ing)?)\s+(?P<object>(?:wafer\s+)?prices?[A-Za-z\s]{{0,20}}?)(?:\s+by\s+{_QUANTITY})?(?:\s+{_TIME})?",
            # "prices fell / dropped / declined"
            rf"(?P<object>[A-Za-z][\w\s]{{1,40}}?)\s+(?:prices?|pricing|cost|ASP)\s+(?P<action>(?:fell|drop|declin|decreas|tumbl)(?:es?|ed|ing|s|ped)?)(?:\s+(?:by\s+)?{_QUANTITY})?(?:\s+{_TIME})?",
            # "X% price increase / decrease"
            rf"(?P<quantity>\d[\d,\.]*\s*%)\s+(?P<object>price)\s+(?P<action>(?:increase|hike|rise|decrease|cut|reduction|drop))",
            # "ASP of X rose / fell"
            rf"(?:ASP|average\s+selling\s+price)\s+(?:of\s+)?(?P<object>[A-Za-z][\w\s]{{1,30}}?)\s+(?P<action>(?:rose|fell|increas|decreas|climb|drop)(?:es?|ed|ing|s|ped)?)\s*(?:(?:by\s+)?{_QUANTITY})?",
            # "pricing power"
            rf"{_ACTOR}\s+(?P<action>(?:gain|los)(?:es?|ed|ing|t)?)\s+(?P<object>pricing\s+power)",
            # "cost per wafer increased"
            rf"(?P<object>cost\s+per\s+(?:wafer|chip|unit|die))\s+(?P<action>(?:increas|rose|jump|climb|decreas|fell|drop)(?:es?|ed|ing|s|ped)?)(?:\s+(?:by\s+)?{_QUANTITY})?",
            # "raised guidance on pricing"
            rf"{_ACTOR}\s+(?P<action>rais(?:es?|ed|ing)?)\s+(?P<object>(?:pricing|price)\s+(?:guidance|forecast|outlook))",
            # "premium pricing for"
            rf"{_ACTOR}\s+(?P<action>(?:commands?|charges?|sets?|demand)(?:ed|ing|s)?)\s+(?P<object>(?:premium|higher)\s+(?:pricing|prices?)\s+for\s+[A-Za-z\s]{{2,30}}?)",
        ],
        "guidance_change": [
            # "NVIDIA raised / lowered guidance"
            rf"{_ACTOR}\s+(?P<action>rais(?:es?|ed|ing)?|lower(?:s|ed|ing)?|cut(?:s|ting)?|updat(?:es?|ed|ing)?|revis(?:es?|ed|ing)?|maintain(?:s|ed|ing)?|reaffirm(?:s|ed|ing)?|reiterat(?:es?|ed|ing)?)\s+(?P<object>(?:(?:revenue|earnings|profit|margin|sales|income)\s+)?(?:guidance|forecast|outlook|estimate|target|expectation)[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "expects revenue of / expects to earn"
            rf"{_ACTOR}\s+(?P<action>expect(?:s|ed|ing)?)\s+(?P<object>(?:revenue|earnings|sales|profit|income|margin)[\w\s]{{0,20}}?)\s+(?:of\s+)?{_QUANTITY}(?:\s+{_TIME})?",
            # "guided for / projects"
            rf"{_ACTOR}\s+(?P<action>guid(?:es?|ed|ing)?\s+for|project(?:s|ed|ing)?|forecast(?:s|ed|ing)?)\s+(?P<object>(?:revenue|sales|earnings)[A-Za-z\s]{{0,20}}?)\s+(?:of\s+)?{_QUANTITY}(?:\s+{_TIME})?",
            # "beat / missed estimates"
            rf"{_ACTOR}\s+(?P<action>(?:beat|miss|exceed|top|surpass|fell\s+short\s+of)(?:es?|ed|ing|s)?)\s+(?P<object>(?:(?:analyst|street|consensus|Wall\s+Street)\s+)?(?:estimates?|expectations?|forecasts?))",
            # "upside / downside surprise"
            rf"{_ACTOR}\s+(?P<action>report(?:s|ed|ing)?)\s+(?:an?\s+)?(?P<object>(?:upside|downside|positive|negative|earnings)\s+(?:surprise|beat|miss))",
            # "revenue came in at / above / below"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>(?:revenue|earnings|sales|profit|income))\s+(?P<action>(?:came\s+in|was|were)\s+(?:at|above|below))\s+{_QUANTITY}",
            # "outlook is / remains"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>(?:revenue|earnings|sales|margin)?\s*outlook)\s+(?P<action>(?:is|remains?|looks?)\s+(?:strong|weak|positive|negative|bullish|bearish|cautious|optimistic|pessimistic))",
            # "warned / warns of"
            rf"{_ACTOR}\s+(?P<action>warn(?:s|ed|ing)?)\s+(?:of\s+)?(?P<object>(?:lower|weak|soft|declining)\s+(?:revenue|earnings|sales|demand|margins?)[A-Za-z\s]{{0,20}}?)(?:\s+{_TIME})?",
            # "quarterly results"
            rf"{_ACTOR}\s+(?P<action>report(?:s|ed|ing)?)\s+(?P<object>(?:Q[1-4]|quarterly|annual)\s+(?:revenue|earnings|results|profit|sales)[A-Za-z\s]{{0,20}}?)\s+(?:of\s+)?{_QUANTITY}",
            # "gross margin expanded / contracted"
            rf"{_ACTOR}(?:'s)?\s+(?P<object>(?:gross|operating|net)?\s*margins?)\s+(?P<action>(?:expand|contract|improv|declin|narrow|widen)(?:es?|ed|ing|s)?)(?:\s+(?:to|by)\s+{_QUANTITY})?",
        ],
    }

    compiled: dict[str, list[re.Pattern[str]]] = {}
    for event_type, pattern_list in patterns.items():
        compiled[event_type] = [
            re.compile(p, re.IGNORECASE) for p in pattern_list
        ]

    return compiled


class PatternExtractor:
    """
    Regex-based event extractor for semiconductor financial text.

    Lazily compiles patterns on first use. Extracts structured EventRecord
    objects from document text using named capture groups.

    Usage:
        extractor = PatternExtractor()
        events = extractor.extract(doc)
    """

    def __init__(
        self,
        config: EventExtractionConfig | None = None,
        normalizer: TimeNormalizer | None = None,
    ):
        self._config = config or EventExtractionConfig()
        self._normalizer = normalizer or TimeNormalizer()
        self._patterns: dict[str, list[re.Pattern[str]]] | None = None
        self._ticker_extractor = None

    @property
    def patterns(self) -> dict[str, list[re.Pattern[str]]]:
        """Lazy-compile patterns on first access."""
        if self._patterns is None:
            self._patterns = _build_patterns()
        return self._patterns

    def _get_ticker_extractor(self):
        """Get TickerExtractor via singleton (avoids circular import)."""
        if self._ticker_extractor is None:
            from src.ingestion.base_adapter import get_ticker_extractor
            self._ticker_extractor = get_ticker_extractor()
        return self._ticker_extractor

    def extract(self, doc: "NormalizedDocument") -> list[EventRecord]:
        """
        Extract events from a document.

        Args:
            doc: NormalizedDocument to extract events from.

        Returns:
            List of EventRecord objects, sorted by span_start.
        """
        text = doc.content
        if not text:
            return []

        events: list[EventRecord] = []
        seen_spans: set[tuple[int, int]] = set()

        for event_type, compiled_patterns in self.patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())

                    # Skip overlapping matches
                    if self._overlaps(span, seen_spans):
                        continue

                    event = self._match_to_event(
                        match=match,
                        event_type=event_type,
                        doc_id=doc.id,
                        full_text=text,
                        doc_tickers=doc.tickers_mentioned,
                    )

                    if event.confidence >= self._config.min_confidence:
                        events.append(event)
                        seen_spans.add(span)

                    if len(events) >= self._config.max_events_per_doc:
                        break

                if len(events) >= self._config.max_events_per_doc:
                    break
            if len(events) >= self._config.max_events_per_doc:
                break

        events.sort(key=lambda e: e.span_start)
        return events

    def _match_to_event(
        self,
        match: re.Match,
        event_type: str,
        doc_id: str,
        full_text: str,
        doc_tickers: list[str],
    ) -> EventRecord:
        """Convert a regex match to an EventRecord."""
        groups = match.groupdict()

        actor = self._clean_capture(groups.get("actor"))
        action = self._clean_capture(groups.get("action")) or match.group(0)
        obj = self._clean_capture(groups.get("object"))
        quantity = self._clean_capture(groups.get("quantity"))
        time_ref_raw = self._clean_capture(groups.get("time_ref"))

        # Normalize time reference
        time_ref = None
        if time_ref_raw:
            time_ref = self._normalizer.normalize(time_ref_raw)

        # Link tickers from context window around the match
        tickers = self._extract_context_tickers(
            full_text, match.start(), match.end(), doc_tickers
        )

        # Compute confidence
        confidence = self._compute_confidence(actor, tickers, quantity)

        return EventRecord(
            doc_id=doc_id,
            event_type=event_type,
            actor=actor,
            action=action,
            object=obj,
            time_ref=time_ref,
            quantity=quantity,
            tickers=tickers,
            confidence=confidence,
            span_start=match.start(),
            span_end=match.end(),
            extractor_version=self._config.extractor_version,
        )

    def _extract_context_tickers(
        self,
        text: str,
        start: int,
        end: int,
        doc_tickers: list[str],
    ) -> list[str]:
        """Extract tickers from a context window around the match span."""
        window = 200
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        context = text[lo:hi]

        try:
            extractor = self._get_ticker_extractor()
            context_tickers = extractor.extract(context)
        except Exception:
            context_tickers = []

        # Merge with doc-level tickers that appear in the actor/object
        matched_text = text[start:end].upper()
        for ticker in doc_tickers:
            if ticker in matched_text and ticker not in context_tickers:
                context_tickers.append(ticker)

        return sorted(set(context_tickers))

    @staticmethod
    def _compute_confidence(
        actor: str | None,
        tickers: list[str],
        quantity: str | None,
    ) -> float:
        """
        Compute confidence score for an extracted event.

        Base: 0.7 for regex match.
        +0.1 if actor is captured.
        +0.1 if at least one ticker is found.
        +0.1 if a quantity is captured.
        Capped at 1.0.
        """
        confidence = 0.7
        if actor:
            confidence += 0.1
        if tickers:
            confidence += 0.1
        if quantity:
            confidence += 0.1
        return min(1.0, confidence)

    @staticmethod
    def _clean_capture(value: str | None) -> str | None:
        """Clean up a captured group value."""
        if value is None:
            return None
        cleaned = value.strip().rstrip(",. ")
        return cleaned if cleaned else None

    @staticmethod
    def _overlaps(
        span: tuple[int, int],
        seen: set[tuple[int, int]],
    ) -> bool:
        """Check if a span overlaps with any previously seen span."""
        s, e = span
        for ss, se in seen:
            if s < se and e > ss:
                return True
        return False

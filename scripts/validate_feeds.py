#!/usr/bin/env python
"""Validate the curated RSS/Atom feed catalog."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from src.config.feed_validation import LiveFeedCheck, validate_live_feed_urls
from src.config.feeds import FEEDS, FeedCatalogIssue, validate_feed_catalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Only run static duplicate/malformed URL checks.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also live-check feeds that are disabled in the static catalog.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds for live feed checks.",
    )
    args = parser.parse_args(argv)

    static_issues = validate_feed_catalog(FEEDS)
    if static_issues:
        _print_static_issues(static_issues)
        return 1

    if args.skip_live:
        print(f"Static feed catalog validation passed for {len(FEEDS)} feeds.")
        return 0

    results = asyncio.run(
        validate_live_feed_urls(
            FEEDS,
            include_disabled=args.include_disabled,
            timeout=args.timeout,
        )
    )
    _print_live_results(results)
    return 1 if any(not result.ok for result in results) else 0


def _print_static_issues(issues: Sequence[FeedCatalogIssue]) -> None:
    print("Static feed catalog validation failed:")
    for issue in issues:
        print(f"- {issue.feed_slug}: {issue.code}: {issue.message}")


def _print_live_results(results: Sequence[LiveFeedCheck]) -> None:
    for result in results:
        status = "ok" if result.ok else "fail"
        detail = f"{result.status_code or '-'} {result.reason}"
        print(
            f"{status:4} {result.feed_slug:32} entries={result.entry_count:<4} "
            f"status={detail} title={result.title!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

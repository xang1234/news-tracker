"""CLI for the extraction eval harness.

Registered into the top-level ``news-tracker`` CLI via
``main.add_command(eval_group)``.
"""

from __future__ import annotations

import click


@click.group("eval")
def eval_group() -> None:
    """Evaluation harnesses for the parsing/extraction layer."""


@eval_group.command("extraction")
@click.option(
    "--golden", default=None, help="Path to a golden-set JSON (defaults to the built-in set)"
)
@click.option(
    "--method",
    type=click.Choice(["rule", "llm"]),
    default="rule",
    help="Extractor to score: 'rule' (regex, default) or 'llm' (needs SCORING_OPENAI_API_KEY)",
)
@click.option("--verbose", is_flag=True, help="Print per-document misses and spurious claims")
def extraction(golden: str | None, method: str, verbose: bool) -> None:
    """Score an extractor against the golden set.

    Example:
        news-tracker eval extraction
        news-tracker eval extraction --method llm --verbose
    """
    from src.eval.extraction import evaluate, make_llm_extractor, rule_extractor
    from src.eval.golden import load_golden_set

    docs = load_golden_set(golden)
    extractor = rule_extractor if method == "rule" else make_llm_extractor()
    report = evaluate(docs, extractor)

    click.echo(f"\nExtraction eval ({method}) — {len(docs)} documents")
    click.echo("-" * 60)
    click.echo(f"  Expected claims:  {report.total_expected}")
    click.echo(f"  Extracted claims: {report.total_extracted}")
    click.echo(f"  True positives:   {report.true_positives}")
    click.echo(f"  Missed (FN):      {report.false_negatives}")
    click.echo(f"  Spurious (FP):    {report.false_positives}")
    click.echo("-" * 60)
    click.echo(f"  Recall:    {report.recall:.3f}")
    click.echo(f"  Precision: {report.precision:.3f}")
    click.echo(f"  F1:        {report.f1:.3f}")

    if verbose:
        for doc in report.per_doc:
            if not (doc.false_negatives or doc.false_positives):
                continue
            click.echo(f"\n  {doc.doc_id}:")
            for fn in doc.false_negatives:
                click.echo(click.style(f"    MISS     {fn}", fg="yellow"))
            for fp in doc.false_positives:
                click.echo(click.style(f"    SPURIOUS {fp}", fg="red"))

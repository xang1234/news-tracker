"""Benchmark embedding and sentiment inference runtimes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from src.embedding.config import EmbeddingConfig
from src.embedding.service import EmbeddingService, ModelType
from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService

SAMPLE_TEXTS = [
    "NVIDIA raised its AI revenue outlook after another quarter of strong data center demand.",
    (
        "TSMC expects advanced packaging capacity to stay tight as AI accelerator "
        "orders continue to climb."
    ),
    (
        "AMD guided to stronger MI300 shipments, but investors remain cautious "
        "about gross margin pressure."
    ),
    (
        "Intel delayed parts of its foundry roadmap, increasing execution risk "
        "for near-term profitability."
    ),
    "Samsung secured new HBM supply commitments as hyperscaler demand remained elevated.",
    (
        "Micron reported better pricing in memory markets, helped by disciplined "
        "supply and AI demand."
    ),
    (
        "ASML said EUV system demand remains healthy even as some customers "
        "moderate legacy node spending."
    ),
    (
        "Broadcom highlighted custom accelerator wins and resilient networking "
        "demand from cloud operators."
    ),
]

LONG_TEXT = " ".join(SAMPLE_TEXTS * 48)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--embedding-onnx-path",
        type=str,
        default=os.getenv("EMBEDDING_ONNX_MODEL_PATH"),
    )
    parser.add_argument(
        "--embedding-minilm-onnx-path",
        type=str,
        default=os.getenv("EMBEDDING_ONNX_MINILM_MODEL_PATH"),
    )
    parser.add_argument(
        "--sentiment-onnx-path",
        type=str,
        default=os.getenv("SENTIMENT_ONNX_MODEL_PATH"),
    )
    return parser


def _max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _durations_summary(samples: list[float], work_units: int = 1) -> dict[str, float]:
    average_ms = statistics.fmean(samples) * 1000
    median_ms = statistics.median(samples) * 1000
    throughput = work_units / statistics.fmean(samples) if samples else 0.0
    return {
        "avg_ms": round(average_ms, 2),
        "median_ms": round(median_ms, 2),
        "throughput_per_sec": round(throughput, 2),
    }


async def _measure_async(iterations: int, coro_factory: Any) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        await coro_factory()
        samples.append(time.perf_counter() - start)
    return samples


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    embedding_config = EmbeddingConfig(
        backend=args.backend,
        device="cpu",
        batch_size=args.batch_size,
        cache_enabled=False,
        onnx_model_path=args.embedding_onnx_path,
        onnx_minilm_model_path=args.embedding_minilm_onnx_path,
    )
    sentiment_config = SentimentConfig(
        backend=args.backend,
        device="cpu",
        batch_size=args.batch_size,
        cache_enabled=False,
        onnx_model_path=args.sentiment_onnx_path,
    )

    embedding_service = EmbeddingService(config=embedding_config)
    sentiment_service = SentimentService(config=sentiment_config)

    batch_texts = SAMPLE_TEXTS[: args.batch_size]

    async def benchmark_embedding_single() -> None:
        await embedding_service.embed_minilm(SAMPLE_TEXTS[0])

    async def benchmark_embedding_batch() -> None:
        await embedding_service.embed_batch(batch_texts, model_type=ModelType.MINILM)

    async def benchmark_embedding_long() -> None:
        await embedding_service.embed_finbert(LONG_TEXT)

    async def benchmark_sentiment_single() -> None:
        await sentiment_service.analyze(SAMPLE_TEXTS[1])

    async def benchmark_sentiment_batch() -> None:
        await sentiment_service.analyze_batch(batch_texts)

    for _ in range(args.warmup_iterations):
        await benchmark_embedding_single()
        await benchmark_embedding_batch()
        await benchmark_embedding_long()
        await benchmark_sentiment_single()
        await benchmark_sentiment_batch()

    embedding_single = await _measure_async(args.iterations, benchmark_embedding_single)
    embedding_batch = await _measure_async(args.iterations, benchmark_embedding_batch)
    embedding_long = await _measure_async(args.iterations, benchmark_embedding_long)
    sentiment_single = await _measure_async(args.iterations, benchmark_sentiment_single)
    sentiment_batch = await _measure_async(args.iterations, benchmark_sentiment_batch)

    result = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "embedding_stats": embedding_service.get_stats(),
        "sentiment_stats": sentiment_service.get_stats(),
        "max_rss_mb": round(_max_rss_mb(), 2),
        "benchmarks": {
            "embedding_minilm_single": _durations_summary(embedding_single),
            "embedding_minilm_batch": _durations_summary(
                embedding_batch,
                work_units=len(batch_texts),
            ),
            "embedding_finbert_long": _durations_summary(embedding_long),
            "sentiment_single": _durations_summary(sentiment_single),
            "sentiment_batch": _durations_summary(
                sentiment_batch,
                work_units=len(batch_texts),
            ),
        },
    }

    await embedding_service.close()
    await sentiment_service.close()
    return result


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""MMSU benchmark."""

# H200 Full-Set Reference Results
# Reproducibility references for the FULL eval set — NOT CI thresholds.
# CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).
# If your PR moves any of these numbers, call it out in the PR description.
#
# Benchmark: MMSU     |  Dataset: MMSU full (5000 samples)
# Hardware:  1× H200 (default; non-H200 sources are tagged in Source column)
# Last verified: 2026-04-17
#
# Accuracy (accuracy)
# | Model      | Config                | overall_accuracy | parseable_samples | unparseable_samples | Source                                           |
# | ---------- | --------------------- | ---------------- | ----------------- | ------------------- | ------------------------------------------------ |
# | Qwen3-Omni | modalities=text       | 72.08%           | 4999/5000         | 1                   | PR #261 [H100, full-set, c=1, max-tokens=32]     |
# TODO(@PopSoda2002): re-run on H200 full-set to replace H100 fallback
# | Qwen3-Omni | modalities=text+audio | TBD              | TBD               | TBD                 | TBD                                              |
# TODO(@PopSoda2002): re-run on H200 — no text+audio full-set run exists
#
# Per-task accuracy (accuracy.per_task; top-level task names only — full sub/sub-sub trees stay in JSON output)
# | Model      | Config                | per_task breakdown (highlights)                                                                                                           | Source                                       |
# | ---------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
# | Qwen3-Omni | modalities=text       | strong: casual_reasoning / polysemy_reasoning / long_speech_summarization = 100%; weak: dialogue_turn_counting 15.15%, volume_comparison 23.64%, pitch_comparison 29.63% | PR #261 [H100, full-set]                     |
# TODO(@PopSoda2002): re-run on H200 — populate full per_task dict
# | Qwen3-Omni | modalities=text+audio | TBD                                                                                                                                       | TBD                                          |
# TODO(@PopSoda2002): re-run on H200 — no text+audio full-set run exists
#
# Speed (speed)
# | Model      | Config                | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source |
# | ---------- | --------------------- | -------------- | ------------- | -------------- | -------------- | ------------- | ------ |
# | Qwen3-Omni | modalities=text       | TBD            | TBD           | TBD            | TBD            | TBD           | TBD    |
# TODO(@PopSoda2002): re-run on H200 — PR #261 did not report latency_mean_s/throughput_qps for the full 5000-sample run
# | Qwen3-Omni | modalities=text+audio | TBD            | TBD           | TBD            | TBD            | TBD           | TBD    |
# TODO(@PopSoda2002): re-run on H200 — no text+audio full-set run exists
# =============================================================================

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.mmsu import (
    build_mmsu_results,
    compute_mmsu_metrics,
    make_mmsu_send_fn,
    print_mmsu_summary,
    save_mmsu_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run(args: argparse.Namespace) -> dict:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"
    modalities = ["text", "audio"] if args.modalities == "text+audio" else ["text"]

    samples = load_mmsu_samples(
        max_samples=args.max_samples,
        task_names=args.task_names.split(",") if args.task_names else None,
        categories=args.categories.split(",") if args.categories else None,
        seed=args.seed,
    )

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn_kwargs = dict(
        modalities=modalities,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_audio_dir=save_audio_dir,
    )
    if args.prompt:
        send_fn_kwargs["prompt"] = args.prompt
    send_fn = make_mmsu_send_fn(args.model, api_url, **send_fn_kwargs)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    results = build_mmsu_results(request_results, samples, modalities)
    metrics = compute_mmsu_metrics(results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    audio_mode = "audio" in modalities

    print_mmsu_summary(metrics, args.model, speed_metrics=speed if audio_mode else None)

    if args.output_dir:
        save_mmsu_results(
            results,
            metrics,
            {
                "model": args.model,
                "base_url": base_url,
                "modalities": modalities,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            args.output_dir,
            speed_metrics=speed if audio_mode else None,
        )

    return {"accuracy": metrics, "speed": speed}


def main() -> None:
    p = argparse.ArgumentParser(description="MMSU benchmark.")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", type=str, default="qwen3-omni")
    p.add_argument("--modalities", choices=["text", "text+audio"], default="text")
    p.add_argument("--output-dir", type=str, default="results/mmsu")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--task-names", type=str, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

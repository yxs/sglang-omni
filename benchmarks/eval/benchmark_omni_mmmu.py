# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark for sglang-omni models.

Evaluates VLM accuracy and performance on the MMMU validation set via
/v1/chat/completions with image input.

Usage:
    # Text-only
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 20

    # With concurrency
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

    # With audio (requires speech server)
    # Note (Yifei, Chenyang): Concurrency=1 only for now since code_predictor and
    # code2wav modules serialize GPU access, so they run serially even when
    # concurrency > 1. And, audio output is still slow at this stage.

    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 5 --enable-audio --max-tokens 50
"""

# H200 Full-Set Reference Results
# Reproducibility references for the FULL eval set — NOT CI thresholds.
# CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).
# If your PR moves any of these numbers, call it out in the PR description.
#
# Benchmark: MMMU     |  Dataset: MMMU_val (900 samples, all 30 subjects)
# Hardware:  1× H200 (default; non-H200 sources are tagged in Source column)
# Last verified: 2026-04-17
#
# Accuracy (summary)
# | Model      | Config             | accuracy | correct | failed | mc_fallback | Source                                  |
# | ---------- | ------------------ | -------- | ------- | ------ | ----------- | --------------------------------------- |
# | Qwen3-Omni | enable_audio=False | 66.44%   | 598/900 | 0      | TBD         | PR #260 (900 samples, max_tokens=2048)  |
# TODO(@PasserBy4): fill mc_fallback — PR #260 body did not report this field
# | Qwen3-Omni | enable_audio=True  | TBD      | TBD     | TBD    | TBD         | TBD                                     |
# TODO(@PasserBy4): re-run on H200 — no enable_audio=True full-900 run exists in any PR
#
# Speed (speed)
# | Model      | Config             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source |
# | ---------- | ------------------ | -------------- | ------------- | -------------- | -------------- | ------------- | ------ |
# | Qwen3-Omni | enable_audio=False | TBD            | TBD           | TBD            | TBD            | TBD           | TBD    |
# TODO(@PasserBy4): re-run on H200
# | Qwen3-Omni | enable_audio=True  | TBD            | TBD           | TBD            | TBD            | TBD           | TBD    |
# TODO(@PasserBy4): re-run on H200

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    SampleOutput,
    calculate_wer_metrics,
    load_asr_model,
    print_speed_summary,
    print_wer_summary,
    transcribe_and_compute_wer,
)
from benchmarks.tasks.visual_understand import (
    compute_mmmu_metrics,
    make_mmmu_send_fn,
    print_mmmu_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MMMUEvalConfig:
    model: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    enable_audio: bool = False
    asr_device: str = "cuda:0"
    lang: str = "en"
    repo_id: str | None = None


def _build_base_url(config: MMMUEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_mmmu_eval(config: MMMUEvalConfig) -> dict:
    """Run full MMMU evaluation and return results dict.

    Returns a dict with keys: summary, speed, config,
    per_sample, and wer (only when enable_audio is True).
    """
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_mmmu_samples(config.max_samples, repo_id=config.repo_id)
    logger.info(f"Prepared {len(samples)} MMMU samples")

    audio_dir: str | None = None
    if config.enable_audio and config.output_dir:
        audio_dir = str(Path(config.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    send_fn = make_mmmu_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        enable_audio=config.enable_audio,
        audio_dir=audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    summary, per_sample = compute_mmmu_metrics(samples, request_results)
    speed_metrics = compute_speed_metrics(
        request_results, wall_clock_s=runner.wall_clock_s
    )

    config_dict = {
        "model": config.model,
        "base_url": base_url,
        "max_samples": config.max_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "max_concurrency": config.max_concurrency,
        "warmup": config.warmup,
        "enable_audio": config.enable_audio,
    }

    results = {
        "summary": summary,
        "speed": speed_metrics,
        "config": config_dict,
        "per_sample": per_sample,
    }

    if config.enable_audio:
        wer_results = _compute_audio_wer(
            request_results, config.lang, config.asr_device
        )
        results["wer"] = wer_results

    if config.output_dir:
        save_json_results(results, config.output_dir, "mmmu_results.json")

    return results


def _compute_audio_wer(
    request_results: list,
    lang: str,
    asr_device: str,
) -> dict:
    """Transcribe audio outputs with ASR and compute WER against text outputs.

    Text output is the reference; ASR transcription of the audio is the
    hypothesis.  Returns a dict with summary and per_sample keys.
    """
    asr = load_asr_model(lang, asr_device)

    outputs: list[SampleOutput] = []
    for result in request_results:

        ref_text = " ".join(result.text.split())
        output = SampleOutput(
            sample_id=result.request_id,
            target_text=ref_text,
            latency_s=result.latency_s,
            audio_duration_s=result.audio_duration_s,
        )

        if not result.is_success or not result.wav_path:
            output.error = result.error or "No audio in response"
            outputs.append(output)
            continue

        output = transcribe_and_compute_wer(
            output, result.wav_path, asr, lang, asr_device
        )
        outputs.append(output)

    wer_summary = calculate_wer_metrics(outputs, lang)

    per_sample = [
        {
            "id": o.sample_id,
            "is_success": o.is_success,
            "wer": o.wer if o.is_success else None,
            "ref_text": o.target_text[:100],
            "hyp_text": o.whisper_text[:100],
            "ref_norm": o.ref_norm,
            "hyp_norm": o.hyp_norm,
            "audio_duration_s": o.audio_duration_s,
            "error": o.error,
        }
        for o in outputs
    ]

    return {"summary": wer_summary, "per_sample": per_sample}


def _config_from_args(args: argparse.Namespace) -> MMMUEvalConfig:
    return MMMUEvalConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        enable_audio=args.enable_audio,
        asr_device=args.asr_device,
        lang=args.lang,
        repo_id=args.repo_id,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_mmmu_eval(config)
    print_mmmu_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"], config.model, config.max_concurrency, title="MMMU Speed"
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMMU benchmark for VLM models served by sglang-omni."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Request audio output and compute text-audio WER.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="Device for ASR model (default: cuda:0).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for ASR transcription (default: en).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo (e.g. 'zhaochenyang20/mmmu-ci-50'). "
        "Defaults to loading the full MMMU/MMMU (all 30 subjects).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/mmmu_audio" if args.enable_audio else "results/mmmu"

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()

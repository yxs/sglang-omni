# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: ``_Code2WavStreamingExecutor`` throughput vs concurrency.

Drives the executor via its public ``add_request`` + ``get_result`` API and
a real ``StreamQueue`` — identical to how the pipeline framework exercises
it in production.  Works on both ``main`` (serialized ``asyncio.Lock``
path) and ``talker-pipeline`` (opportunistic micro-batch path) with no
script-side changes, so the two JSON outputs are directly comparable.

Typical workflow for the #276 Phase 1 PR:

    # 1. Checkout main, run once
    git checkout main
    pip install --no-deps -e .
    python benchmarks/perf/bench_code2wav.py \
        --model-path /models/Qwen3-Omni-30B-A3B-Instruct \
        --gpu-id 0 \
        --concurrencies 1 4 8 16 \
        --output-json results_main.json

    # 2. Checkout the optimized branch, run again
    git checkout talker-pipeline
    pip install --no-deps -e .
    python benchmarks/perf/bench_code2wav.py \
        --model-path /models/Qwen3-Omni-30B-A3B-Instruct \
        --gpu-id 0 \
        --concurrencies 1 4 8 16 \
        --output-json results_phase1.json

    # 3. Diff the two JSON files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_executor import (
    _Code2WavStreamingExecutor,
    load_code2wav_model,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import OmniRequest, StagePayload

logger = logging.getLogger("bench_code2wav")


@dataclass
class LevelResult:
    """Per-concurrency-level summary.

    Metric naming intentionally follows vllm-omni PR #1246 / vllm bench
    serve conventions so that sglang-omni data is directly comparable.
    """

    concurrency: int
    num_requests: int
    total_wall_s: float
    # Per-request end-to-end latency (equivalent to vllm-omni's E2EL)
    per_request_latency_ms_mean: float
    per_request_latency_ms_p50: float
    per_request_latency_ms_p95: float
    per_request_latency_ms_p99: float
    # Request throughput = num_requests / wall_time
    throughput_req_per_s: float
    # Audio duration synthesized in total (s); fixed by config, not measured
    audio_duration_s: float
    # Audio Real-Time Factor = wall_time / audio_duration; <1.0 = faster than realtime
    audio_rtf: float
    # Audio throughput = audio_duration / wall_time (vllm-omni AUDIO_THROUGHPUT)
    audio_throughput_s_per_s: float


async def _drive_one(
    exe: _Code2WavStreamingExecutor,
    stream_queue: StreamQueue,
    request_id: str,
    codes_chunks: list[torch.Tensor],
) -> float:
    """Submit one request end-to-end via the executor's public API.

    Opens a stream queue for the request, fires ``add_request``, feeds all
    codes chunks through the stream queue, signals done, then awaits the
    executor's internal per-request task.  Returns wall time in ms.
    """
    stream_queue.open(request_id)
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=""),
        data=None,
    )

    start = time.perf_counter()
    await exe.add_request(payload)
    for i, codes in enumerate(codes_chunks):
        stream_queue.put(
            request_id,
            StreamItem(
                chunk_id=i,
                data=codes,
                from_stage="bench",
            ),
        )
    stream_queue.put_done(request_id)

    # exe._tasks is how the executor tracks per-request work internally;
    # awaiting it gives a direct per-request wall-time measurement that
    # avoids the shared ``get_result`` pool's completion-order ambiguity.
    task = exe._tasks[request_id]
    await task
    return (time.perf_counter() - start) * 1000.0


async def run_level(
    exe: _Code2WavStreamingExecutor,
    stream_queue: StreamQueue,
    *,
    concurrency: int,
    num_requests: int,
    chunks_per_request: int,
    num_quantizers: int,
    codebook_size: int,
    device: torch.device,
    request_seq_start: int,
    total_upsample: int,
    sample_rate: int,
    var_length: bool,
) -> tuple[LevelResult, int]:
    """Fire ``num_requests`` requests with the given concurrency cap."""
    sem = asyncio.Semaphore(concurrency)

    def _chunk_count(req_idx: int) -> int:
        """Deterministic per-request chunk count.

        Uniform ``chunks_per_request`` by default; in ``--var-length`` mode
        draws from ``[chunks_per_request//2, chunks_per_request]`` so the
        batched path actually exercises padded positions.
        """
        if not var_length:
            return chunks_per_request
        lo = max(1, chunks_per_request // 2)
        # Use the request index itself as a cheap deterministic sample.
        return lo + (req_idx * 7) % (chunks_per_request - lo + 1)

    async def one(req_idx: int) -> float:
        async with sem:
            # Distinct random codes per request (deterministic via seed
            # for a given request index, for reproducibility).  ``high``
            # matches the code2wav codebook size — any id >= codebook_size
            # would trigger an OOB embedding lookup at forward time.
            g = torch.Generator(device="cpu").manual_seed(req_idx)
            n_chunks = _chunk_count(req_idx)
            chunks = [
                torch.randint(
                    low=0,
                    high=codebook_size,
                    size=(num_quantizers,),
                    dtype=torch.long,
                    generator=g,
                ).to(device=device)
                for _ in range(n_chunks)
            ]
            return await _drive_one(
                exe, stream_queue, f"bench-{req_idx}", chunks,
            )

    next_seq = request_seq_start

    # Per-level warmup: fire ``concurrency`` parallel requests so MIOpen /
    # cuDNN prime the actual batched shape that the measurement window
    # will hit.  The executor's own ``__init__`` warmup already primed
    # {1, 2, 4, 8, max_batch_size} with fixed shapes, but this catches
    # any per-level-shape variation (especially under --var-length).
    warmup_start = next_seq
    next_seq += concurrency
    await asyncio.gather(*(one(warmup_start + i) for i in range(concurrency)))

    start = time.perf_counter()
    latencies = await asyncio.gather(
        *(one(next_seq + i) for i in range(num_requests))
    )
    wall = time.perf_counter() - start
    next_seq += num_requests

    latencies.sort()

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    # Audio duration is deterministic from the config:
    # each request produces ``chunks_per_request * total_upsample`` output
    # samples (using the average under --var-length).  Dividing by
    # ``sample_rate`` gives seconds per request.
    if var_length:
        avg_chunks = (chunks_per_request // 2 + chunks_per_request) / 2
    else:
        avg_chunks = chunks_per_request
    audio_per_req_s = avg_chunks * total_upsample / sample_rate
    audio_total_s = audio_per_req_s * num_requests

    return (
        LevelResult(
            concurrency=concurrency,
            num_requests=num_requests,
            total_wall_s=wall,
            per_request_latency_ms_mean=statistics.fmean(latencies),
            per_request_latency_ms_p50=_pct(latencies, 50),
            per_request_latency_ms_p95=_pct(latencies, 95),
            per_request_latency_ms_p99=_pct(latencies, 99),
            throughput_req_per_s=num_requests / wall if wall > 0 else 0.0,
            audio_duration_s=audio_total_s,
            audio_rtf=wall / audio_total_s if audio_total_s > 0 else 0.0,
            audio_throughput_s_per_s=audio_total_s / wall if wall > 0 else 0.0,
        ),
        next_seq,
    )


async def main_async(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")

    logger.info("Loading code2wav weights from %s ...", args.model_path)
    model = load_code2wav_model(
        args.model_path,
        device=str(device),
        dtype=args.dtype,
    )

    # The executor expects ``stream_chunk_size`` to equal the size of a
    # decode window.  We feed exactly one chunk per chunk so the first
    # decode fires after ``stream_chunk_size`` pushes.  ``left_context_size``
    # matches the production default (25) so measured numbers reflect the
    # actual vocoder workload (T = chunks_per_request + left_context).
    exe = _Code2WavStreamingExecutor(
        model,
        device=str(device),
        stream_chunk_size=args.chunks_per_request,
        left_context_size=args.left_context_size,
    )
    stream_queue = StreamQueue()
    exe._stream_queue = stream_queue

    results: list[LevelResult] = []
    request_seq = 0
    for concurrency in args.concurrencies:
        logger.info(
            "=== concurrency=%d, requests=%d, chunks=%d, Q=%d, var_length=%s ===",
            concurrency,
            args.requests_per_level,
            args.chunks_per_request,
            args.num_quantizers,
            args.var_length,
        )
        lr, request_seq = await run_level(
            exe,
            stream_queue,
            concurrency=concurrency,
            num_requests=args.requests_per_level,
            chunks_per_request=args.chunks_per_request,
            num_quantizers=args.num_quantizers,
            codebook_size=args.codebook_size,
            device=device,
            request_seq_start=request_seq,
            total_upsample=exe._total_upsample,
            sample_rate=exe._sample_rate,
            var_length=args.var_length,
        )
        results.append(lr)
        logger.info(
            "  wall=%.3fs  mean=%.1fms  p50=%.1fms  p95=%.1fms  p99=%.1fms  "
            "tput=%.2f req/s  rtf=%.3f  audio_tput=%.2f s/s",
            lr.total_wall_s,
            lr.per_request_latency_ms_mean,
            lr.per_request_latency_ms_p50,
            lr.per_request_latency_ms_p95,
            lr.per_request_latency_ms_p99,
            lr.throughput_req_per_s,
            lr.audio_rtf,
            lr.audio_throughput_s_per_s,
        )

    # Shutdown any background tasks the optimized executor may have started
    task = getattr(exe, "_batch_loop_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    payload: dict[str, Any] = {
        "model_path": args.model_path,
        "device": str(device),
        "chunks_per_request": args.chunks_per_request,
        "num_quantizers": args.num_quantizers,
        "left_context_size": args.left_context_size,
        "var_length": args.var_length,
        "results": [asdict(r) for r in results],
    }
    if args.output_json:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True
        )
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results written to %s", args.output_json)
    else:
        print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--gpu-id", type=int, default=0, help="Set -1 for CPU")
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--requests-per-level", type=int, default=32)
    p.add_argument(
        "--chunks-per-request", type=int, default=10,
        help="Number of time-step chunks fed per request (executor's "
             "stream_chunk_size is set to this so one decode fires)",
    )
    p.add_argument("--num-quantizers", type=int, default=16)
    p.add_argument(
        "--codebook-size", type=int, default=2048,
        help="Upper bound (exclusive) for random codes.  Must not exceed "
             "the model's actual codebook_size or forward will OOB.",
    )
    p.add_argument(
        "--left-context-size", type=int, default=25,
        help="Left-context frames per decode window (matches the production "
             "default).  Set to 0 to benchmark the kernel without context.",
    )
    p.add_argument(
        "--var-length", action="store_true",
        help="Randomize chunks per request in [chunks_per_request//2, "
             "chunks_per_request] so the batched path exercises padded "
             "positions instead of always running max_len == seq_len.",
    )
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

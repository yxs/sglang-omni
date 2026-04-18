# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: ``_CodePredictorStreamingExecutor`` throughput vs concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.components.code_predictor_executor import (
    _CodePredictorStreamingExecutor,
    _CodePredictorWrapper,
    _load_talker_model,
)
from sglang_omni.proto import OmniRequest, StagePayload

logger = logging.getLogger("bench_code_predictor")


@dataclass
class _FakeStreamItem:
    data: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class _FakeStreamQueue:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[_FakeStreamItem | None]] = {}

    def open(self, request_id: str) -> None:
        self._queues.setdefault(request_id, asyncio.Queue())

    def put(self, request_id: str, item: _FakeStreamItem) -> None:
        self._queues.setdefault(request_id, asyncio.Queue()).put_nowait(item)

    def put_done(self, request_id: str) -> None:
        self._queues.setdefault(request_id, asyncio.Queue()).put_nowait(None)

    async def get(self, request_id: str) -> _FakeStreamItem | None:
        return await self._queues[request_id].get()

    def close(self, request_id: str) -> None:
        q = self._queues.get(request_id)
        if q is not None:
            q.put_nowait(None)


@dataclass
class LevelResult:
    concurrency: int
    num_requests: int
    chunks_per_request: int
    total_wall_s: float
    per_request_latency_ms_mean: float
    per_request_latency_ms_p50: float
    per_request_latency_ms_p95: float
    per_request_latency_ms_p99: float
    throughput_req_per_s: float
    throughput_chunks_per_s: float


async def _drive_one(
    exe: _CodePredictorStreamingExecutor,
    stream_queue: _FakeStreamQueue,
    request_id: str,
    chunks: list[_FakeStreamItem],
) -> float:
    stream_queue.open(request_id)
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=""),
        data=None,
    )
    exe.set_stream_fn(lambda *args, **kwargs: None)

    start = time.perf_counter()
    task = asyncio.create_task(exe.add_request(payload))
    for chunk in chunks:
        stream_queue.put(request_id, chunk)
    stream_queue.put_done(request_id)
    await task
    return (time.perf_counter() - start) * 1000.0


async def run_level(
    exe: _CodePredictorStreamingExecutor,
    stream_queue: _FakeStreamQueue,
    *,
    concurrency: int,
    num_requests: int,
    chunks_per_request: int,
    hidden_size: int,
    codebook_size: int,
    device: torch.device,
    request_seq_start: int,
    dtype: torch.dtype,
) -> tuple[LevelResult, int]:
    sem = asyncio.Semaphore(concurrency)

    async def one(req_idx: int) -> float:
        async with sem:
            g = torch.Generator(device="cpu").manual_seed(req_idx)
            chunks: list[_FakeStreamItem] = []
            for _ in range(chunks_per_request):
                h = torch.randn(
                    hidden_size,
                    generator=g,
                    dtype=torch.float32,
                ).to(device=device, dtype=dtype)
                c = int(torch.randint(0, codebook_size, (), generator=g))
                chunks.append(_FakeStreamItem(data=h, metadata={"codec_code": c}))
            return await _drive_one(
                exe,
                stream_queue,
                f"bench-{req_idx}",
                chunks,
            )

    next_seq = request_seq_start

    warmup_start = next_seq
    next_seq += concurrency
    await asyncio.gather(*(one(warmup_start + i) for i in range(concurrency)))

    start = time.perf_counter()
    latencies = await asyncio.gather(*(one(next_seq + i) for i in range(num_requests)))
    wall = time.perf_counter() - start
    next_seq += num_requests
    latencies.sort()

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    total_chunks = num_requests * chunks_per_request

    return (
        LevelResult(
            concurrency=concurrency,
            num_requests=num_requests,
            chunks_per_request=chunks_per_request,
            total_wall_s=wall,
            per_request_latency_ms_mean=statistics.fmean(latencies),
            per_request_latency_ms_p50=_pct(latencies, 50),
            per_request_latency_ms_p95=_pct(latencies, 95),
            per_request_latency_ms_p99=_pct(latencies, 99),
            throughput_req_per_s=num_requests / wall if wall > 0 else 0.0,
            throughput_chunks_per_s=total_chunks / wall if wall > 0 else 0.0,
        ),
        next_seq,
    )


async def main_async(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    logger.info("Loading code predictor from %s ...", args.model_path)
    talker_shell = _load_talker_model(args.model_path, gpu_id=args.gpu_id)
    wrapper = _CodePredictorWrapper(talker_shell)
    hidden_size = talker_shell.config.text_config.hidden_size
    vocab_size = talker_shell.config.text_config.vocab_size
    logger.info(
        "Loaded: hidden_size=%d  vocab=%d  num_code_groups=%d  device=%s",
        hidden_size,
        vocab_size,
        talker_shell.config.num_code_groups,
        device,
    )

    exe = _CodePredictorStreamingExecutor(
        model=wrapper,
        device=str(device),
        max_batch_size=args.max_batch_size,
    )

    stream_queue = _FakeStreamQueue()
    exe._stream_queue = stream_queue

    results: list[LevelResult] = []
    request_seq = 0
    for concurrency in args.concurrencies:
        logger.info(
            "=== concurrency=%d, requests=%d, chunks_per_request=%d ===",
            concurrency,
            args.requests_per_level,
            args.chunks_per_request,
        )
        lr, request_seq = await run_level(
            exe,
            stream_queue,
            concurrency=concurrency,
            num_requests=args.requests_per_level,
            chunks_per_request=args.chunks_per_request,
            hidden_size=hidden_size,
            codebook_size=vocab_size,
            device=device,
            request_seq_start=request_seq,
            dtype=dtype,
        )
        results.append(lr)
        logger.info(
            "  wall=%.3fs  mean=%.1fms  p50=%.1fms  p95=%.1fms  p99=%.1fms  "
            "tput=%.2f req/s  %.1f chunks/s",
            lr.total_wall_s,
            lr.per_request_latency_ms_mean,
            lr.per_request_latency_ms_p50,
            lr.per_request_latency_ms_p95,
            lr.per_request_latency_ms_p99,
            lr.throughput_req_per_s,
            lr.throughput_chunks_per_s,
        )

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
        "dtype": args.dtype,
        "chunks_per_request": args.chunks_per_request,
        "max_batch_size": args.max_batch_size,
        "results": [asdict(r) for r in results],
    }
    if args.output_json:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_json)) or ".",
            exist_ok=True,
        )
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results written to %s", args.output_json)
    else:
        print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--requests-per-level", type=int, default=16)
    p.add_argument("--chunks-per-request", type=int, default=10)
    p.add_argument("--max-batch-size", type=int, default=16)
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

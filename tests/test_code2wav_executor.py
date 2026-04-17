# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the code2wav executor's micro-batch scheduler.

These tests use a small deterministic mock vocoder so they run on CPU
without any model weights.  The mock satisfies the contract required
by ``_Code2WavStreamingExecutor``:

* ``total_upsample`` attribute.
* Forward on ``codes [B, Q, T]`` returns ``wav [B, 1, T * U]``.

We verify three things:

1. ``_vocoder_forward`` produces the expected waveform for a single
   ``[Q, T]`` window (shape, values, trim honoured).
2. ``_forward_batch`` returns per-request audio identical to running
   each request through ``_vocoder_forward`` individually — padding
   must not leak into the valid region for a mock whose output at
   position t only depends on ``codes[:, :, t]``.
3. The async ``_batch_decode_step`` path dispatches results through
   the attached ``asyncio.Future`` correctly: a lone request takes the
   fast path; multiple requests are batched; aborted requests are
   cancelled without triggering a forward.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_omni.components.code2wav_executor import (
    _Code2WavStreamingExecutor,
    _DecodeRequest,
)


class _MockVocoder(nn.Module):
    """Deterministic mock vocoder with per-position output.

    For input ``codes [B, Q, T]`` the output at batch ``i``, position
    ``t * U + u`` is ``codes[i, :, t].sum() + u * 0.001``.  Crucially,
    each output sample depends only on the codes at the *same* time
    position, so padding tokens at the tail never contaminate the
    valid region — this lets us assert exact equality between the
    batched path and the single-request path.
    """

    def __init__(self, upsample: int = 4) -> None:
        super().__init__()
        self.total_upsample = int(upsample)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.dim() == 3, "expect [B, Q, T]"
        _, _, T = codes.shape
        per_position = codes.float().sum(dim=1)                       # [B, T]
        upsampled = per_position.repeat_interleave(
            self.total_upsample, dim=1,
        )                                                              # [B, T*U]
        offset = torch.arange(
            T * self.total_upsample, device=codes.device, dtype=torch.float32,
        ) * 0.001
        return (upsampled + offset).unsqueeze(1)                       # [B, 1, T*U]


def _make_executor(
    *,
    upsample: int = 4,
    max_batch_size: int = 4,
    stream_chunk_size: int = 10,
    left_context_size: int = 0,
) -> _Code2WavStreamingExecutor:
    vocoder = _MockVocoder(upsample=upsample)
    return _Code2WavStreamingExecutor(
        vocoder,
        device="cpu",
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        max_batch_size=max_batch_size,
    )


# ----------------------------------------------------------------------
# Synchronous forward-path tests
# ----------------------------------------------------------------------


def test_vocoder_forward_produces_expected_waveform() -> None:
    exe = _make_executor(upsample=4)
    # Q=2, T=3 → per-position sums = [1+4, 2+5, 3+6] = [5, 7, 9]
    # Mock offset is global sample index, so samples 0..11 get offsets 0..11.
    codes = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    wav = exe._vocoder_forward(codes, trim_samples=0)

    expected = np.array(
        [
            5.000, 5.001, 5.002, 5.003,
            7.004, 7.005, 7.006, 7.007,
            9.008, 9.009, 9.010, 9.011,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(wav, expected, rtol=1e-5)


def test_vocoder_forward_honours_trim_samples() -> None:
    exe = _make_executor(upsample=4)
    codes = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    wav = exe._vocoder_forward(codes, trim_samples=4)

    # First position (4 samples) are trimmed away
    expected = np.array(
        [
            7.004, 7.005, 7.006, 7.007,
            9.008, 9.009, 9.010, 9.011,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(wav, expected, rtol=1e-5)


def test_forward_batch_matches_independent_single_forwards() -> None:
    """Batched path must produce identical per-request audio as single path."""
    exe = _make_executor(upsample=4)
    # Three requests with different lengths AND different trims.
    #   - req0: Q=1, T=3, no context
    #   - req1: Q=1, T=5, 1 position of context (trim 4 samples)
    #   - req2: Q=1, T=2, no context
    codes_list = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
        torch.tensor([[9, 10]], dtype=torch.long),
    ]
    trims = [0, 4, 0]

    requests = [
        _DecodeRequest(
            request_id=f"r{i}",
            codes_window=codes,
            trim_samples=trim,
        )
        for i, (codes, trim) in enumerate(zip(codes_list, trims))
    ]

    batch_outputs = exe._forward_batch(requests)

    assert len(batch_outputs) == 3
    for i, (codes, trim) in enumerate(zip(codes_list, trims)):
        single = exe._vocoder_forward(codes, trim_samples=trim)
        assert batch_outputs[i].shape == single.shape, (
            f"req{i}: batch shape {batch_outputs[i].shape} != single {single.shape}"
        )
        np.testing.assert_allclose(
            batch_outputs[i], single, rtol=1e-5,
            err_msg=f"req{i}: batched output diverges from single-path output",
        )


# ----------------------------------------------------------------------
# Async scheduling path tests
# ----------------------------------------------------------------------


async def _shutdown_batch_loop(exe: _Code2WavStreamingExecutor) -> None:
    if exe._batch_loop_task is None:
        return
    exe._batch_loop_task.cancel()
    try:
        await exe._batch_loop_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_single_pending_request_takes_fast_path() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        req = _DecodeRequest(
            request_id="r0",
            codes_window=codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._pending.put_nowait(req)

        wav = await req.result_future

        expected = exe._vocoder_forward(codes, trim_samples=0)
        np.testing.assert_allclose(wav, expected, rtol=1e-5)
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_multiple_pending_requests_are_batched() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes_list = [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
            torch.tensor([[9, 10]], dtype=torch.long),
        ]
        reqs = [
            _DecodeRequest(
                request_id=f"r{i}",
                codes_window=c,
                trim_samples=0,
                result_future=loop.create_future(),
            )
            for i, c in enumerate(codes_list)
        ]
        # Enqueue all three with no await in between so the batch loop
        # sees them in a single drain.
        for r in reqs:
            exe._pending.put_nowait(r)

        results = await asyncio.gather(*(r.result_future for r in reqs))

        assert len(results) == 3
        for i, (got, codes) in enumerate(zip(results, codes_list)):
            expected = exe._vocoder_forward(codes, trim_samples=0)
            np.testing.assert_allclose(
                got, expected, rtol=1e-5,
                err_msg=f"req{i}: batched dispatch diverges from single-path",
            )
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_aborted_request_is_cancelled() -> None:
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        req = _DecodeRequest(
            request_id="r0",
            codes_window=codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._aborted.add("r0")
        exe._pending.put_nowait(req)

        with pytest.raises(asyncio.CancelledError):
            await req.result_future
    finally:
        await _shutdown_batch_loop(exe)


@pytest.mark.asyncio
async def test_mixed_batch_live_plus_aborted() -> None:
    """A live request in the same batch as an aborted one should still succeed."""
    exe = _make_executor()
    exe._ensure_batch_loop()
    try:
        loop = asyncio.get_running_loop()
        live_codes = torch.tensor([[1, 2, 3]], dtype=torch.long)
        live = _DecodeRequest(
            request_id="live",
            codes_window=live_codes,
            trim_samples=0,
            result_future=loop.create_future(),
        )
        aborted = _DecodeRequest(
            request_id="dead",
            codes_window=torch.tensor([[9, 9, 9]], dtype=torch.long),
            trim_samples=0,
            result_future=loop.create_future(),
        )
        exe._aborted.add("dead")
        exe._pending.put_nowait(live)
        exe._pending.put_nowait(aborted)

        wav = await live.result_future
        with pytest.raises(asyncio.CancelledError):
            await aborted.result_future

        expected = exe._vocoder_forward(live_codes, trim_samples=0)
        np.testing.assert_allclose(wav, expected, rtol=1e-5)
    finally:
        await _shutdown_batch_loop(exe)

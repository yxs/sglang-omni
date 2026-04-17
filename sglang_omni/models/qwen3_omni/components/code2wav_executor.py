# SPDX-License-Identifier: Apache-2.0
"""Code2Wav executor with incremental waveform streaming.

Supports opportunistic micro-batching: when multiple requests have pending
vocoder decode work simultaneously, their windows are padded to a common
length and forwarded in a single batched GPU call.  When only one request
is active the fast-path runs exactly the same logic as before (no padding
overhead).  See ``_batch_decode_loop`` for the scheduling strategy.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang_omni.executors import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH_SIZE: int = 16


@dataclass
class _DecodeRequest:
    """A single vocoder decode request submitted by a per-request task."""

    request_id: str
    codes_window: torch.Tensor  # [num_quantizers, seq_len]
    trim_samples: int
    result_future: asyncio.Future[np.ndarray] = field(default=None)  # type: ignore[assignment]


def load_code2wav_model(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    weight_prefix: str = "code2wav.",
):
    """Load Code2Wav model from HF checkpoint with the given weight prefix."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = getattr(config, "code2wav_config", None)
    if code2wav_config is None:
        raise ValueError(f"No code2wav_config found in {model_path}")

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)

    model = load_module(
        model,
        model_path,
        prefix=weight_prefix,
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    return model


class _Code2WavStreamingExecutor(Executor):
    """Decode codec chunks incrementally and emit audio stream chunks.

    GPU work is scheduled through an opportunistic micro-batch loop rather
    than a global ``asyncio.Lock``.  Per-request tasks submit
    ``_DecodeRequest`` objects and ``await`` the attached Future; the batch
    loop drains all pending requests, pads them to a common sequence length,
    and runs a single batched ``self._model(codes)`` forward pass.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        stream_chunk_size: int = 10,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._sample_rate = sample_rate
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(getattr(model, "total_upsample", 1))
        self._max_batch_size = max(int(max_batch_size), 1)
        self._stream_queue: Any | None = None
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._stream_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._aborted: set[str] = set()

        # Micro-batch scheduling state (replaces the old _gpu_lock)
        self._pending: asyncio.Queue[_DecodeRequest] = asyncio.Queue()
        self._batch_loop_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Executor interface
    # ------------------------------------------------------------------

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        self._ensure_batch_loop()
        self._stream_queues[request_id] = asyncio.Queue()
        task = asyncio.create_task(self._run_request(payload))
        self._tasks[request_id] = task
        task.add_done_callback(lambda _task: self._done.put_nowait(request_id))

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            if request_id in self._aborted:
                self._stream_queues.pop(request_id, None)
                continue
            try:
                return await task
            except Exception as exc:
                exc.request_id = request_id
                raise

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
        queue = self._stream_queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(None)

    async def stream(self, request_id: str):
        queue = self._stream_queues.get(request_id)
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    # ------------------------------------------------------------------
    # Per-request streaming loop (unchanged control flow)
    # ------------------------------------------------------------------

    async def _run_request(self, payload: StagePayload) -> StagePayload:
        request_id = payload.request_id
        if self._stream_queue is None:
            raise RuntimeError("Code2Wav executor requires a stream queue")

        queue = self._stream_queues[request_id]
        code_chunks: list[torch.Tensor] = []
        audio_chunks: list[np.ndarray] = []
        emitted_positions = 0

        try:
            while True:
                if request_id in self._aborted:
                    raise asyncio.CancelledError()

                item = await self._stream_queue.get(request_id)
                if item is None:
                    break

                codes = item.data.to(device=self._device, dtype=torch.long)
                if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
                    continue
                code_chunks.append(codes)
                ready_positions = len(code_chunks) - emitted_positions
                if ready_positions < self._stream_chunk_size:
                    continue

                audio = await self._submit_decode(
                    request_id, code_chunks, emitted_positions, len(code_chunks),
                )
                emitted_positions = len(code_chunks)
                if audio.size == 0:
                    continue
                audio_chunks.append(audio)
                await queue.put(self._build_audio_payload(audio))

            if code_chunks and emitted_positions < len(code_chunks):
                audio = await self._submit_decode(
                    request_id, code_chunks, emitted_positions, len(code_chunks),
                )
                if audio.size > 0:
                    audio_chunks.append(audio)
                    await queue.put(self._build_audio_payload(audio))

            await queue.put(None)

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
            else:
                full_audio = np.zeros((0,), dtype=np.float32)

            payload.data = self._build_audio_payload(full_audio)
            return payload
        finally:
            self._stream_queues.pop(request_id, None)

    # ------------------------------------------------------------------
    # Decode submission (replaces _decode_async + _gpu_lock)
    # ------------------------------------------------------------------

    def _prepare_window(
        self,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, int]:
        """Build the codes window tensor and compute trim length.

        Returns:
            (codes_window [Q, T], trim_samples)
        """
        context_size = min(self._left_context_size, start_index)
        window = torch.stack(
            code_chunks[start_index - context_size : end_index], dim=0,
        )
        codes_window = window.transpose(0, 1)  # [Q, T]
        trim_samples = context_size * self._total_upsample
        return codes_window, trim_samples

    async def _submit_decode(
        self,
        request_id: str,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        """Prepare the codes window and submit to the batch loop."""
        if start_index >= end_index:
            return np.zeros((0,), dtype=np.float32)

        codes_window, trim_samples = self._prepare_window(
            code_chunks, start_index, end_index,
        )

        if self._device.type == "cpu":
            return self._vocoder_forward(codes_window, trim_samples)

        loop = asyncio.get_running_loop()
        req = _DecodeRequest(
            request_id=request_id,
            codes_window=codes_window,
            trim_samples=trim_samples,
            result_future=loop.create_future(),
        )
        self._pending.put_nowait(req)
        return await req.result_future

    # ------------------------------------------------------------------
    # Opportunistic micro-batch loop
    # ------------------------------------------------------------------

    def _ensure_batch_loop(self) -> None:
        if self._batch_loop_task is None or self._batch_loop_task.done():
            self._batch_loop_task = asyncio.create_task(self._batch_decode_loop())

    async def _batch_decode_loop(self) -> None:
        """Drain pending decode requests and forward them in batches.

        Strategy (opportunistic / scheme B):
          1. Block until at least one request arrives.
          2. Immediately drain everything else currently queued.
          3. Forward the whole batch on GPU.
          4. Dispatch per-request results back via Futures.
          5. Repeat.

        Single-request fast path avoids padding overhead entirely.

        The outer loop catches non-cancellation exceptions so that a single
        failed batch (e.g. CUDA OOM) does not kill the loop and deadlock
        all subsequent callers waiting on their Futures.
        """
        while True:
            try:
                await self._batch_decode_step()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("batch decode loop iteration failed")

    async def _batch_decode_step(self) -> None:
        """Execute one gather-and-forward cycle."""
        loop = asyncio.get_running_loop()

        first = await self._pending.get()
        batch: list[_DecodeRequest] = [first]

        while (
            not self._pending.empty()
            and len(batch) < self._max_batch_size
        ):
            try:
                batch.append(self._pending.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Filter out aborted requests before touching GPU
        live = [r for r in batch if r.request_id not in self._aborted]
        aborted = [r for r in batch if r.request_id in self._aborted]
        for r in aborted:
            if not r.result_future.done():
                r.result_future.cancel()
        if not live:
            return

        try:
            if len(live) == 1:
                r = live[0]
                audio = await loop.run_in_executor(
                    None, self._vocoder_forward,
                    r.codes_window, r.trim_samples,
                )
                # NOTE: set_result must be called from the event loop thread.
                # _vocoder_forward runs in a thread pool, so we return the
                # result here rather than setting it inside the worker.
                if not r.result_future.done():
                    r.result_future.set_result(audio)
            else:
                audios = await loop.run_in_executor(
                    None, self._forward_batch, live,
                )
                for req, audio in zip(live, audios):
                    if not req.result_future.done():
                        req.result_future.set_result(audio)
        except Exception as exc:
            for req in live:
                if not req.result_future.done():
                    req.result_future.set_exception(exc)
            raise

    # ------------------------------------------------------------------
    # GPU forward paths (run in thread pool)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _vocoder_forward(
        self,
        codes_window: torch.Tensor,
        trim_samples: int,
    ) -> np.ndarray:
        """Run the vocoder on a single ``[Q, T]`` codes window.

        Used by the single-request fast path and the CPU fallback.
        The batched path uses ``_forward_batch`` which inlines similar
        trimming logic per-request on the batched output.
        """
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        codes = codes_window.unsqueeze(0)  # [1, Q, T]
        wav = self._model(codes)
        if trim_samples:
            wav = wav[..., trim_samples:]
        return wav.reshape(-1).detach().cpu().float().numpy().copy()

    @torch.no_grad()
    def _forward_batch(
        self, batch: list[_DecodeRequest],
    ) -> list[np.ndarray]:
        """Batch path: pad to common length, forward once, split results."""
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        max_len = max(r.codes_window.shape[1] for r in batch)
        num_q = batch[0].codes_window.shape[0]

        # Pad with codec_eos_token_id rather than 0.  Zero is a valid
        # codebook entry whose embedding would produce audible artifacts
        # in the CNN vocoder's receptive field near the valid/padding
        # boundary.  EOS is a training-time sentinel that produces
        # near-silence, making boundary leakage benign.
        padded = torch.full(
            (len(batch), num_q, max_len),
            fill_value=self._codec_eos_token_id,
            dtype=batch[0].codes_window.dtype,
            device=self._device,
        )
        for i, req in enumerate(batch):
            seq_len = req.codes_window.shape[1]
            padded[i, :, :seq_len] = req.codes_window

        wav_batch = self._model(padded)  # [B, 1, waveform_len]

        results: list[np.ndarray] = []
        for i, req in enumerate(batch):
            wav = wav_batch[i]
            valid_samples = req.codes_window.shape[1] * self._total_upsample
            if req.trim_samples:
                wav = wav[..., req.trim_samples:valid_samples]
            else:
                wav = wav[..., :valid_samples]
            results.append(wav.reshape(-1).detach().cpu().float().numpy().copy())

        return results

    def _build_audio_payload(self, audio: np.ndarray) -> dict[str, Any]:
        audio = audio.astype(np.float32, copy=False)
        return {
            "audio_waveform": audio.tobytes(),
            "audio_waveform_shape": list(audio.shape),
            "audio_waveform_dtype": "float32",
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }


def create_code2wav_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
) -> Executor:
    """Create Code2Wav executor that streams waveform chunks."""
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return _Code2WavStreamingExecutor(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        max_batch_size=max_batch_size,
    )

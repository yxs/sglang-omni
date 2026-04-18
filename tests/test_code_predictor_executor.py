# SPDX-License-Identifier: Apache-2.0
"""Parity + abort tests for ``_CodePredictorStreamingExecutor``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.components.code_predictor_executor import (
    _CodePredictorStreamingExecutor,
    _CodePredictorWrapper,
    _LightweightTalkerShell,
)

_TINY_HIDDEN_SIZE = 64
_TINY_NUM_HEADS = 4
_TINY_NUM_KV_HEADS = 2
_TINY_NUM_LAYERS = 2
_TINY_INTERMEDIATE = 128
_TINY_VOCAB = 256
_TINY_NUM_CODE_GROUPS = 4


@pytest.fixture(scope="module")
def tiny_predictor_config() -> Qwen3OmniMoeTalkerCodePredictorConfig:
    return Qwen3OmniMoeTalkerCodePredictorConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN_SIZE,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=_TINY_NUM_LAYERS,
        num_attention_heads=_TINY_NUM_HEADS,
        num_key_value_heads=_TINY_NUM_KV_HEADS,
        num_code_groups=_TINY_NUM_CODE_GROUPS,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
    )


@pytest.fixture(scope="module")
def tiny_talker_shell(
    tiny_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig,
) -> _LightweightTalkerShell:
    torch.manual_seed(0)

    predictor = (
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
            tiny_predictor_config,
        )
    )
    predictor = predictor.to(dtype=torch.float32).eval()

    codec_embedding = nn.Embedding(_TINY_VOCAB, _TINY_HIDDEN_SIZE).eval()

    @dataclass
    class _TalkerCfgStub:
        num_code_groups: int

    shell = _LightweightTalkerShell(
        config=_TalkerCfgStub(num_code_groups=_TINY_NUM_CODE_GROUPS),
        code_predictor=predictor,
        codec_embedding=codec_embedding,
    )
    shell.eval()
    return shell


@pytest.fixture(scope="module")
def tiny_wrapper(tiny_talker_shell: _LightweightTalkerShell) -> _CodePredictorWrapper:
    return _CodePredictorWrapper(tiny_talker_shell)


def _random_chunk_inputs(
    num_requests: int,
    hidden_size: int,
    vocab: int,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_requests):
        h = torch.randn(hidden_size, generator=g)
        c = torch.randint(0, vocab, (), dtype=torch.long, generator=g)
        inputs.append((h, c))
    return inputs


@pytest.mark.parametrize("num_requests", [1, 2, 4, 8])
def test_batched_wrapper_matches_single_request_path(
    tiny_wrapper: _CodePredictorWrapper,
    num_requests: int,
) -> None:
    inputs = _random_chunk_inputs(
        num_requests,
        _TINY_HIDDEN_SIZE,
        _TINY_VOCAB,
    )

    singles: list[dict[str, torch.Tensor]] = []
    for h, c in inputs:
        with torch.no_grad():
            out = tiny_wrapper(
                talker_hidden=h.unsqueeze(0),
                layer0_code=c.unsqueeze(0),
            )
        singles.append(
            {
                "codes": out["codes"][0].clone(),
                "summed_embeddings": out["summed_embeddings"][0].clone(),
            }
        )

    talker_hidden = torch.stack([h for h, _ in inputs], dim=0)
    layer0_code = torch.stack([c for _, c in inputs], dim=0)
    with torch.no_grad():
        batched = tiny_wrapper(
            talker_hidden=talker_hidden,
            layer0_code=layer0_code,
        )

    for i in range(num_requests):
        assert torch.equal(
            batched["codes"][i], singles[i]["codes"]
        ), f"request {i}: codes mismatch"
        assert torch.allclose(
            batched["summed_embeddings"][i],
            singles[i]["summed_embeddings"],
            atol=1e-5,
            rtol=1e-5,
        ), f"request {i}: summed_embeddings mismatch"


def test_wrapper_output_shapes_and_dtypes(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    h = torch.randn(3, _TINY_HIDDEN_SIZE)
    c = torch.randint(0, _TINY_VOCAB, (3,), dtype=torch.long)
    with torch.no_grad():
        out = tiny_wrapper(talker_hidden=h, layer0_code=c)

    assert out["codes"].shape == (3, _TINY_NUM_CODE_GROUPS)
    assert out["codes"].dtype == torch.long
    assert out["summed_embeddings"].shape == (3, _TINY_HIDDEN_SIZE)
    assert out["summed_embeddings"].dtype == torch.float32


def test_wrapper_first_code_is_layer0_code(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    h = torch.randn(5, _TINY_HIDDEN_SIZE)
    c = torch.randint(0, _TINY_VOCAB, (5,), dtype=torch.long)
    with torch.no_grad():
        out = tiny_wrapper(talker_hidden=h, layer0_code=c)
    assert torch.equal(out["codes"][:, 0], c)


class _FakeStreamItem:
    def __init__(self, data: torch.Tensor, codec_code: int) -> None:
        self.data = data
        self.metadata = {"codec_code": codec_code}


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


@pytest.mark.asyncio
async def test_abort_stops_dispatch(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    stream_queue = _FakeStreamQueue()
    exe._stream_queue = stream_queue

    dispatched: list[tuple[str, Any]] = []

    def capture(request_id: str, tensor: torch.Tensor, *, target_stage: str) -> None:
        dispatched.append((request_id, target_stage))

    exe.set_stream_fn(capture)

    rid = "req-abort"
    stream_queue.open(rid)

    from sglang_omni.proto import OmniRequest, StagePayload

    payload = StagePayload(
        request_id=rid,
        request=OmniRequest(inputs=""),
        data=None,
    )

    task = asyncio.create_task(exe.add_request(payload))

    stream_queue.put(
        rid, _FakeStreamItem(data=torch.randn(_TINY_HIDDEN_SIZE), codec_code=1)
    )
    for _ in range(3):
        await asyncio.sleep(0)

    await exe.abort(rid)

    stream_queue.put(
        rid, _FakeStreamItem(data=torch.randn(_TINY_HIDDEN_SIZE), codec_code=2)
    )
    stream_queue.put_done(rid)

    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        raise AssertionError("add_request did not return after abort")

    assert all(r == rid for r, _ in dispatched)

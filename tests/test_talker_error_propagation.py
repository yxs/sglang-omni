# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming_omni.components.talker_executor import MingTalkerExecutor
from sglang_omni.models.qwen3_omni.components import talker_executor as qwen3_te
from sglang_omni.proto import StagePayload


@pytest.mark.asyncio
async def test_ming_talker_propagates_oom(monkeypatch) -> None:
    executor = MingTalkerExecutor(model_path="/fake/model/path")
    monkeypatch.setattr(executor, "_extract_text", lambda payload: "hello world")

    def _boom(text: str) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(executor, "_generate_speech", _boom)
    payload = StagePayload(request_id="t1", request=None, data={})

    with pytest.raises(torch.cuda.OutOfMemoryError):
        await executor.add_request(payload)


def test_qwen3_talker_propagates_oom(monkeypatch) -> None:
    obj = qwen3_te.TalkerStreamingExecutor.__new__(qwen3_te.TalkerStreamingExecutor)
    obj._tts_special_cache = None
    obj._tts_bos_token_id = 0
    obj._tts_eos_token_id = 1
    obj._tts_pad_token_id = 2
    obj._resolved_model_path = "/fake/model/path"
    obj._device = "cpu"
    obj._dtype = torch.float32

    def _boom(model_path: str, row_ids: list[int]) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(qwen3_te, "_load_thinker_embedding_rows", _boom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        obj._get_tts_special_embeds()

# SPDX-License-Identifier: Apache-2.0
"""Regression tests for talker executor OOM propagation.

Covers both Ming-Omni and Qwen3-Omni talker TTS paths, ensuring CUDA
OOM errors surface to callers instead of being swallowed into silent
None-waveform fallbacks.

Reference: https://github.com/sgl-project/sglang-omni/issues/300

Author:
Xuesong Ye https://github.com/yxs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang_omni.models.ming_omni.components.talker_executor import MingTalkerExecutor
from sglang_omni.models.qwen3_omni.components import talker_executor as qwen3_te
from sglang_omni.proto import OmniRequest, StagePayload

_OOM_MSG = "CUDA out of memory (injected)"


@pytest.mark.asyncio
async def test_ming_talker_propagates_oom() -> None:
    executor = MingTalkerExecutor(model_path="/fake/model/path")
    payload = StagePayload(
        request_id="t1",
        request=MagicMock(spec=OmniRequest),
        data={},
    )

    with (
        patch.object(executor, "_extract_text", return_value="hello world"),
        patch.object(
            executor,
            "_generate_speech",
            side_effect=torch.OutOfMemoryError(_OOM_MSG),
        ),
        pytest.raises(torch.OutOfMemoryError),
    ):
        await executor.add_request(payload)


def test_qwen3_talker_propagates_oom() -> None:
    executor = MagicMock(spec=qwen3_te.TalkerStreamingExecutor)
    executor._tts_special_cache = None
    executor._tts_bos_token_id = 0
    executor._tts_eos_token_id = 1
    executor._tts_pad_token_id = 2
    executor._resolved_model_path = "/fake/model/path"

    with (
        patch.object(
            qwen3_te,
            "_load_thinker_embedding_rows",
            side_effect=torch.OutOfMemoryError(_OOM_MSG),
        ),
        pytest.raises(torch.OutOfMemoryError),
    ):
        qwen3_te.TalkerStreamingExecutor._get_tts_special_embeds(executor)

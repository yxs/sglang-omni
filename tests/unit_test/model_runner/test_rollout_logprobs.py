# SPDX-License-Identifier: Apache-2.0
"""Rollout logprob recording uses sampler-computed logprobs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner


def test_rollout_logprobs_record_sampler_values_not_raw_logits() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.tensor([[2.0, 1.0]])
    token_ids = torch.tensor([0])
    raw_logprob = torch.log_softmax(logits, dim=-1)[0, 0].item()
    sampler_logprob = torch.log_softmax(logits / 0.5, dim=-1)[0, 0].item()
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    request = SimpleNamespace(data=data)

    runner._record_rollout_logprobs(
        torch.tensor([sampler_logprob]), token_ids, [request]
    )

    assert len(data.output_token_logprobs) == 1
    assert data.output_token_logprobs[0][1] == 0
    assert math.isclose(data.output_token_logprobs[0][0], sampler_logprob, abs_tol=1e-4)
    assert not math.isclose(data.output_token_logprobs[0][0], raw_logprob, abs_tol=1e-4)


def test_rollout_logprobs_align_per_request_at_batch_2() -> None:
    runner = object.__new__(ModelRunner)
    data0 = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    data1 = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    requests = [SimpleNamespace(data=data0), SimpleNamespace(data=data1)]

    runner._record_rollout_logprobs(
        torch.tensor([-0.5, -1.5]), torch.tensor([11, 22]), requests
    )

    # each request must get ITS OWN row, not the other request's token/logprob
    assert data0.output_token_logprobs[0][1] == 11
    assert data1.output_token_logprobs[0][1] == 22
    assert math.isclose(data0.output_token_logprobs[0][0], -0.5, abs_tol=1e-4)
    assert math.isclose(data1.output_token_logprobs[0][0], -1.5, abs_tol=1e-4)


def test_rollout_logprobs_skips_on_batch_size_mismatch() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    # more logprobs than requests => batching assumption broke => skip, not take first N
    runner._record_rollout_logprobs(
        torch.tensor([-0.5, -1.5]), torch.tensor([11, 22]), [SimpleNamespace(data=data)]
    )

    assert data.output_token_logprobs == []

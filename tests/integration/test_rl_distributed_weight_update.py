# SPDX-License-Identifier: Apache-2.0
"""End-to-end distributed weight-update (refit) test for the RL admin control plane.

Mirrors sglang's ``test/registered/rl/test_update_weights_from_distributed.py``,
adapted to the SGLang-Omni multi-stage pipeline. A rank-0 "trainer" process
broadcasts base-model weights over a NCCL process group; the omni ``tts_engine``
stage receives them through the admin control plane
(``/init_weights_update_group`` + ``/update_weights_from_distributed``). The test
asserts the SHA256 weights checksum changes (the refit actually happened) and the
server still serves audio afterwards, and prints the refit latency.

Only the non-tied ``body.*`` transformer params are broadcast: tied audio-codebook
weights require trainer-side ``convert_to_hf`` mapping, which is the training side's
concern, not the inference-side control plane under test here.

Requires 2 GPUs + the Higgs TTS-4B (instruct) and 4B-base checkpoints in the HF
cache. Skipped otherwise (e.g. in CPU/1-GPU CI).

Validated on 2xH100 (sglang 0.5.8): update success, checksum changed, audio served,
latency ~0.58s for all 397 body params.

Run manually on a 2-GPU box::

    python -m pytest tests/integration/test_rl_distributed_weight_update.py -s
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import pytest
import requests

HF_TTS_MODEL = "bosonai/higgs-audio-v3-tts-4b"
BASE_CACHE_DIRNAME = "models--boson-sglang--higgs-audio-v3-generation-4B-base"
SERVER_PORT = int(os.environ.get("OMNI_E2E_PORT", "8200"))
MASTER_PORT = int(os.environ.get("OMNI_E2E_MASTER_PORT", "29570"))
GROUP_NAME = "rl_e2e_weight_update_group"
URL = f"http://localhost:{SERVER_PORT}"


def _hf_hub_root() -> str:
    return os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )


def _resolve_base_dir() -> str | None:
    snaps = glob.glob(
        os.path.join(_hf_hub_root(), BASE_CACHE_DIRNAME, "snapshots", "*")
    )
    for snap in snaps:
        if glob.glob(os.path.join(snap, "*.safetensors")):
            return snap
    return None


def _two_gpus() -> bool:
    try:
        import torch

        return torch.cuda.device_count() >= 2
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_two_gpus() and _resolve_base_dir()),
    reason="requires 2 GPUs + Higgs 4B-base checkpoint in the HF cache",
)


# --------------------------------------------------------------------------- #
# rank-0 trainer: broadcasts base-model body weights over NCCL.
# Invoked as a subprocess: `python <thisfile> trainer <port> <group> <base_dir>
#                            <prefix> <limit> <manifest_path>`
# --------------------------------------------------------------------------- #
def _run_trainer() -> None:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    import torch
    from safetensors.torch import load_file

    try:
        from sglang.srt.utils import init_custom_process_group
    except Exception:
        from sglang.srt.utils.common import init_custom_process_group

    _, _, port, group, base_dir, prefix, limit, manifest_path = sys.argv
    port, limit = int(port), int(limit)

    torch.cuda.set_device(0)
    state: dict[str, "torch.Tensor"] = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        state.update(load_file(f, device="cuda:0"))
    names = sorted(n for n in state if n.startswith(prefix))
    if limit > 0:
        names = names[:limit]
    manifest = {
        "names": names,
        "dtypes": [str(state[n].dtype).replace("torch.", "") for n in names],
        "shapes": [list(state[n].shape) for n in names],
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)
    print(f"[trainer] MANIFEST_WRITTEN n={len(names)}", flush=True)

    group_handle = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        world_size=2,
        rank=0,
        group_name=group,
    )
    torch.cuda.synchronize()
    print("[trainer] RENDEZVOUS_OK", flush=True)
    for n in names:
        torch.distributed.broadcast(state[n], src=0, group=group_handle)
    torch.cuda.synchronize()
    print("[trainer] BROADCAST_DONE", flush=True)
    try:
        torch.distributed.destroy_process_group(group_handle)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Test fixtures / helpers
# --------------------------------------------------------------------------- #
def _wait_for(predicate, timeout: float, interval: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="module")
def omni_server():
    """Boot a Higgs TTS-4B omni server on GPU 1 with NCCL env matched to the trainer."""
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="1",
        NCCL_CUMEM_ENABLE="0",
        NCCL_NVLS_ENABLE="0",
        HF_HUB_OFFLINE=env.get("HF_HUB_OFFLINE", "1"),
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            HF_TTS_MODEL,
            "--port",
            str(SERVER_PORT),
        ],
        env=env,
    )

    def _ready() -> bool:
        try:
            return requests.get(f"{URL}/model_info", timeout=5).status_code == 200
        except Exception:
            return False

    if not _wait_for(_ready, timeout=600):
        proc.kill()
        pytest.fail("omni server did not become ready within 600s")
    yield URL
    proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def _tts_checksum() -> dict:
    r = requests.post(
        f"{URL}/weights_checker", json={"action": "checksum"}, timeout=180
    )
    r.raise_for_status()
    for res in r.json().get("results", []):
        if res.get("stage") == "tts_engine":
            return res.get("data", {})
    return {}


def test_distributed_refit_changes_weights_and_keeps_serving(omni_server, tmp_path):
    base_dir = _resolve_base_dir()
    manifest = str(tmp_path / "manifest.json")
    trainer_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1")
    trainer = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "trainer",
            str(MASTER_PORT),
            GROUP_NAME,
            base_dir,
            "body.",
            "0",
            manifest,
        ],
        env=trainer_env,
    )
    try:
        assert _wait_for(
            lambda: os.path.exists(manifest), timeout=180
        ), "trainer never wrote manifest"
        time.sleep(4)
        spec = json.load(open(manifest))

        before = _tts_checksum().get("per_gpu_checksum")
        assert before, "no baseline checksum from tts_engine stage"

        r = requests.post(
            f"{URL}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": MASTER_PORT,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": GROUP_NAME,
                "backend": "nccl",
                "stages": ["tts_engine"],
            },
            timeout=180,
        )
        assert r.status_code == 200 and r.json()["success"], r.text

        t0 = time.perf_counter()
        r = requests.post(
            f"{URL}/update_weights_from_distributed",
            json={
                "names": spec["names"],
                "dtypes": spec["dtypes"],
                "shapes": spec["shapes"],
                "group_name": GROUP_NAME,
                "stages": ["tts_engine"],
            },
            timeout=600,
        )
        latency = time.perf_counter() - t0
        assert r.status_code == 200 and r.json()["success"], r.text
        print(
            f"\n[refit] {len(spec['names'])} params updated in {latency:.3f}s",
            flush=True,
        )

        after = _tts_checksum().get("per_gpu_checksum")
        assert after and after != before, "weights checksum did not change after refit"

        audio = requests.post(
            f"{URL}/v1/audio/speech",
            json={"input": "After distributed refit."},
            timeout=180,
        )
        assert (
            audio.status_code == 200 and len(audio.content) > 1000
        ), "server stopped serving audio after refit"
    finally:
        try:
            trainer.wait(timeout=120)
        except Exception:
            trainer.kill()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trainer":
        _run_trainer()
    else:
        raise SystemExit("run via pytest; 'trainer' subcommand is internal")

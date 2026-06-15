# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response protocol definitions."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class UsageResponse(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionAudio(BaseModel):
    """Audio data returned in a chat completion response."""

    id: str
    data: str  # base64-encoded audio
    expires_at: int | None = None
    transcript: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Streaming
    stream: bool = False

    # Multi-modal output control
    modalities: list[str] | None = None  # e.g. ["text", "audio"]

    # Audio output configuration
    audio: dict[str, Any] | None = None  # {"voice": "...", "format": "wav"}

    # Audio input (sglang-omni extension)
    # Can be a list of audio file paths (local paths or URLs)
    audios: list[str] | None = None

    # Image input (sglang-omni extension)
    # Can be a list of image file paths (local paths or URLs)
    images: list[str] | None = None

    # Video input (sglang-omni extension)
    # Can be a list of video file paths (local paths or URLs)
    videos: list[str] | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None

    # Per-stage sampling overrides (sglang-omni specific)
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None

    # Talker-specific overrides for Qwen3-Omni speech output
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_repetition_penalty: float | None = None
    talker_max_new_tokens: int | None = None

    # Misc
    request_id: str | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageResponse | None = None


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    audio: ChatCompletionAudio | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageResponse | None = None


# ---------------------------------------------------------------------------
# RL Rollout — POST /generate (Miles-compatible, #780 §1.1.1)
# ---------------------------------------------------------------------------


class RolloutGenerateRequest(BaseModel):
    """Miles-compatible RL rollout request for ``POST /generate``.

    This endpoint follows the normal Miles/SGLang AR interface. It is separate
    from the OpenAI-style serving endpoints and does not change their schemas.
    Exactly one prompt input form (``input_ids``, ``prompt``, ``messages``)
    must be supplied.
    """

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None

    # Exactly one prompt input form.
    input_ids: list[int] | None = None  # Miles-compatible alias for prompt_token_ids
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None

    # Generation params.
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False  # default false for RL rollout
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None
    output_modalities: list[str] | None = None  # e.g. ["audio"] or ["text", "audio"]

    # Opaque rollout-side identifiers (rollout_id, group_id, ...). Echoed back
    # in meta_info.request_metadata; not used for scheduling.
    metadata: dict[str, Any] | None = None

    # Rollout artifact controls.
    return_logprob: bool = True  # default true for RL rollout
    return_omni_rollout: bool = True  # default true for Miles-Omni rollout
    return_routed_experts: bool = False
    return_indexer_topk: bool = False


class GenerateFinishReason(BaseModel):
    """Finish status for a rollout generation."""

    type: str  # "stop" | "length" | "abort" | "error"
    length: int | None = None


class GenerateAudio(BaseModel):
    """Audio payload for a rollout generation."""

    data: str | None = None  # base64 audio
    path: str | None = None  # worker-visible/shared-fs audio_ref
    format: str | None = None  # e.g. "wav", "pcm"
    sample_rate: int | None = None


class GenerateMetaInfo(BaseModel):
    """Rollout meta_info block (#780 §1.1.1)."""

    finish_reason: GenerateFinishReason
    prompt_tokens: int | None = None
    completion_tokens: int = 0  # primary generated length
    cached_tokens: int | None = None
    weight_version: str = ""
    request_metadata: dict[str, Any] | None = None  # echo of opaque request metadata
    # Each item is (logprob, token_id) and may include token text as a later field.
    output_token_logprobs: list[Any] | None = None
    # SGLang-Omni multimodal trainable actions (populated by the §1.1.2 container).
    omni_rollout: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate`` (#780 §1.1.1)."""

    text: str = ""
    audio: GenerateAudio | None = None
    meta_info: GenerateMetaInfo


# ---------------------------------------------------------------------------
# Speech (TTS)
# ---------------------------------------------------------------------------


SUPPORTED_TTS_RESPONSE_FORMATS = frozenset({"wav", "mp3", "flac", "pcm", "aac", "opus"})
SUPPORTED_TTS_LANGUAGES = frozenset(
    {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
)
SUPPORTED_TTS_TASK_TYPES = frozenset({"Base", "CustomVoice", "VoiceDesign"})
TTS_SPEED_MIN = 0.25
TTS_SPEED_MAX = 4.0


class SpeechReference(BaseModel):
    """Reference item for voice cloning in /v1/audio/speech."""

    audio_path: str | None = None
    ref_audio: str | None = None
    audio: str | None = None
    data: str | None = None
    media_type: str | None = None
    text: str | None = None
    vq_codes: list[list[int]] | list[int] | None = None


class CreateSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request.

    Standard OpenAI fields plus extensions for advanced TTS models
    (e.g. voice cloning, style instructions).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Standard OpenAI fields
    model: str | None = None
    input: str
    voice: str = Field(
        default="default",
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False

    # Advanced TTS extensions
    task_type: str | None = None  # e.g. "Base", "CustomVoice", "VoiceDesign"
    language: str | None = None
    instructions: str | None = None  # style/emotion instructions

    # Voice cloning parameters
    ref_audio: str | None = None  # path or URL to reference audio
    ref_text: str | None = None  # transcript of reference audio
    references: list[SpeechReference] | None = None  # S2-Pro-style refs
    x_vector_only_mode: bool | None = None
    token_count: int | None = None  # MOSS-TTS duration token target
    duration_tokens: int | None = None  # alias for token_count
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    # Generation parameters
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    # Per-stage overrides (sglang-omni specific)
    stage_params: dict[str, dict[str, Any]] | None = None


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


class ModelPermission(BaseModel):
    """Model permission info."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True


class ModelCard(BaseModel):
    """A single model entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sglang-omni"
    permission: list[ModelPermission] = Field(
        default_factory=lambda: [ModelPermission()]
    )
    root: str | None = None


class ModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Administrative APIs
# ---------------------------------------------------------------------------


class AdminRequestBase(BaseModel):
    """Common admin request routing controls."""

    stages: list[str] | None = None
    timeout_s: float | None = None


class PauseGenerationRequest(AdminRequestBase):
    mode: str = "abort"


class ContinueGenerationRequest(AdminRequestBase):
    torch_empty_cache: bool = True


class UpdateWeightFromDiskRequest(AdminRequestBase):
    model_path: str
    load_format: str | None = None
    abort_all_requests: bool = False
    weight_version: str | None = None
    is_async: bool = False
    torch_empty_cache: bool = False
    keep_pause: bool = False
    recapture_cuda_graph: bool = False
    token_step: int = 0
    flush_cache: bool = True
    manifest: dict[str, Any] | None = None


class UpdateWeightsFromTensorRequest(AdminRequestBase):
    serialized_named_tensors: list[Any] | None = None
    load_format: str | None = None
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: str | None = None
    disable_draft_model: bool | None = None
    torch_empty_cache: bool = False


class UpdateWeightsFromDistributedRequest(AdminRequestBase):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: str | None = None
    load_format: str | None = None
    torch_empty_cache: bool = False


class WeightsCheckerRequest(AdminRequestBase):
    action: str = "checksum"

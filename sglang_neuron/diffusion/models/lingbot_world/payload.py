# SPDX-License-Identifier: Apache-2.0
"""LingBot-World's payload schema between the DiT and its role workers.

The encode role sends the DiT a prompt payload (``prompt_embeds``) once per prompt and one
condition payload per chunk; the DiT sends the decode role each chunk's latents, and the decode
role returns its frames. Every payload carries :func:`chunk_metadata` (the session, the chunk
index, or :data:`PROMPT_CHUNK` for the prompt). Also the request ``condition_inputs`` keys.

Standard library only: the spawned role workers import it before they pin their cores.
"""

from __future__ import annotations

#: Request ``condition_inputs`` keys (the Neuron realtime adapter sets the first two on chunk 0).
CAMERA_TRAJECTORY = "camera_trajectory"
TOTAL_CHUNKS = "neuron_total_chunks"
CAMERA_ACTIONS = "camera_actions"
#: SGLang's realtime adapter sets it on a chunk whose request carries a new prompt
#: (``sglang.multimodal_gen.runtime.realtime.lingbot_world.LINGBOT_PROMPT_UPDATED_CONDITION``).
PROMPT_UPDATED = "lingbot_prompt_updated"

#: Encode -> DiT payload keys (the prompt payload, then one condition payload per chunk).
PROMPT_EMBEDS = "prompt_embeds"
IMAGE_CONDITION = "lingbot_image_condition"
CONDITIONING_METADATA = "lingbot_conditioning_metadata"

#: DiT -> decode payload keys: the chunk's de-normalized VAE latents and their metadata.
LATENTS = "latents"
LATENTS_METADATA = "lingbot_latents_metadata"

#: Decode output keys: the chunk's uint8 frames ``(F, H, W, 3)`` and their metadata.
VIDEO = "video"
VIDEO_METADATA = "lingbot_video_metadata"

#: Chunk id of the once-per-request prompt payload.
PROMPT_CHUNK = "prompt"


def chunk_metadata(
    session_id: str, chunk: int | str, *, last: bool = False, **extra: object
) -> dict[str, object]:
    """The metadata every LingBot payload carries."""
    return {"session_id": str(session_id), "chunk_index": chunk, "last": bool(last), **extra}


#: UMT5's prompt length (tokens; the prompt embedding is ``[1, 512, 4096]``).
MAX_SEQUENCE_LENGTH = 512


def tokenize(tokenizer, text: str, max_length: int = MAX_SEQUENCE_LENGTH):
    """vLLM-Omni's UMT5 tokenization of one prompt (whitespace collapsed, padded)."""
    return tokenizer(
        [" ".join(text.strip().split())],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

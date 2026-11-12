# SPDX-License-Identifier: Apache-2.0
"""SGLang's FLUX.3 text encoder (Qwen3-VL's language model) with its forward as one compiled graph per pad bucket."""

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import Flux3TextEncoder

from sglang_neuron.diffusion.compile_backend import compile_graph
from sglang_neuron.diffusion.flux3_action.dit import DIT_COMPILER_ARGS


class NeuronFlux3TextEncoder(Flux3TextEncoder):
    """``Flux3TextEncoder`` whose context is computed by ``_context`` compiled at fixed shapes (one graph per pad
    bucket of ``pad_multiple`` tokens); prompts are tokenized on the host as SGLang does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = None

    def _context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``hidden_states[k]`` for ``k`` in ``output_layers`` of the language model on one right-padded prompt,
        concatenated: the layers, causal mask and text positions of ``Qwen3VLModel.forward`` on text input."""
        from transformers.masking_utils import create_causal_mask

        model = self.model.language_model
        hidden = model.embed_tokens(input_ids)
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, 1, -1).expand(4, hidden.shape[0], -1)
        mask = create_causal_mask(config=model.config, inputs_embeds=hidden, attention_mask=attention_mask,
                                  past_key_values=None, position_ids=positions[0])
        embeddings = model.rotary_emb(hidden, positions[1:])
        states = []
        for layer in model.layers:
            states.append(hidden)
            hidden = layer(hidden, attention_mask=mask, position_ids=positions[0], past_key_values=None,
                           position_embeddings=embeddings)
        states.append(model.norm(hidden))
        return torch.cat([states[k] for k in self.output_layers], dim=-1).to(self.embed_dtype)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[torch.Tensor]:
        """``Flux3TextEncoder.encode``: contexts ``(1, L_i, len(output_layers) * hidden)``, one prompt per graph
        call."""
        if self._graph is None:
            self._graph = compile_graph(self._context, DIT_COMPILER_ARGS)
        tokenizer = self.processor.tokenizer
        results = []
        for text in texts:
            prompt = self.processor.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                        add_generation_prompt=True)
            toks = tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True,
                             max_length=self._bucket(prompt), padding_side="right")
            results.append(self._graph(toks["input_ids"].to(self.device), toks["attention_mask"].to(self.device)))
        return results

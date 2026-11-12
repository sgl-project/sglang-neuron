# SPDX-License-Identifier: Apache-2.0
"""Keep sampled-token futures compatible across decode graph buckets."""

import torch
from torch import nn
from torch.nn import functional as F


class FixedDecodeTokenIO(nn.Module):
    """Adapt a sampled decode graph to a fixed token-vector capacity.

    Metadata retains the graph's chosen batch bucket. Only token I/O has a
    stable width, so a request's device future remains usable when another DP rank
    changes the group's batch bucket. PendingSample and TokenRelay retain the real
    row count and request identities separately from this transport capacity.
    """

    def __init__(self, model, width, assemble=True, samples=True):
        """``assemble``: build the input ids from the previous forward's device tokens (off on a pipeline stage after
        the first, which takes the batch's placeholder ids as they are). ``samples``: sample and return tokens at the
        fixed width (off on a pipeline stage before the last, which returns its boundary)."""
        super().__init__()
        if width < 1:
            raise ValueError("Decode token width must be positive")
        self.model = model
        self.width = width
        self.assemble, self.samples = bool(assemble), bool(samples)

    def forward(self, input_ids, positions, token_control=None, token_ids_out=None, **kwargs):
        """``token_ids_out`` (int32 [width]): a pipelined decode's last stage samples into this seam token slot in place
        and returns it as its tokens."""
        batch = positions.shape[0]
        if not 0 < batch <= self.width:
            raise ValueError("Decode metadata exceeds the token capacity")
        if self.assemble:
            input_ids = self._assembled(input_ids, token_control)[:batch]
        return self._sampled(self.model(input_ids=input_ids, positions=positions, **kwargs), batch, token_ids_out)

    def _assembled(self, input_ids, token_control):
        """The width's input ids, from ``input_ids``, the previous decode forward's device token vector (or a
        placeholder), and ``token_control``, int32 [2, width]: row 0 the host token ids, row 1 each row's source in
        ``input_ids``, or -1 to take row 0. The ids are assembled inside the graph, so a batch change never makes the
        host read a pending device future, which would block the scheduler for most of a step and hold every DP rank
        at the next collective."""
        if input_ids.ndim != 1 or input_ids.shape[0] != self.width:
            raise ValueError("Decode token input does not match the fixed width")
        if (token_control is None or tuple(token_control.shape) != (2, self.width) or token_control.dtype != torch.int32
                or input_ids.dtype != torch.int32):
            raise ValueError("Reordered decode requires int32 token_control of shape [2, width]")
        source = token_control[1]
        gathered = torch.index_select(input_ids, 0, source.clamp(min=0).to(torch.int64))
        return torch.where(source >= 0, gathered, token_control[0])

    def _sampled(self, result, batch, token_ids_out=None):
        """The graph's sampled tokens at the fixed width, or a stage's boundary as it is."""
        if not self.samples:
            if token_ids_out is not None:
                raise ValueError("a decode graph that samples nothing writes no token slot")
            return result
        if not isinstance(result, dict) or "token_ids" not in result:
            raise TypeError("Fixed decode token I/O requires sampled dictionary outputs")
        tokens = result["token_ids"]
        if tokens.ndim != 1 or tokens.shape[0] != batch or tokens.dtype != torch.int32:
            raise ValueError("Sampled token output does not match the decode batch")
        if token_ids_out is not None:
            if tuple(token_ids_out.shape) != (self.width,) or token_ids_out.dtype != torch.int32:
                raise ValueError(f"the token slot is {tuple(token_ids_out.shape)} {token_ids_out.dtype}, not int32 "
                                 f"[{self.width}]")
            token_ids_out.copy_(tokens if batch == self.width else F.pad(tokens, (0, self.width - batch), value=0))
            return {**result, "token_ids": token_ids_out}
        if batch == self.width:
            return result
        result = dict(result)
        result["token_ids"] = F.pad(tokens, (0, self.width - batch), value=0)
        return result

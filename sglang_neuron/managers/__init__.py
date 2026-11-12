# SPDX-License-Identifier: Apache-2.0
"""Hooks on SGLang's managers: the host trace of the scheduler loop (host_loop), the DP identity and control fence of
a DP-attention decode (dp_identity), and two hooks for aborted requests (logprob_errors)."""

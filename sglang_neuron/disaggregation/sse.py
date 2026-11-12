# SPDX-License-Identifier: Apache-2.0
"""One line of an SGLang server's event stream (``data: {...}``), read by the PD load balancer, which watches a streamed
prefill's own stream and the decode's first event until the decode streams its first token
(balancer._Attempt.first_event)."""

from __future__ import annotations

import json
from typing import Optional, Tuple

DONE = b"data: [DONE]"


def _parse(line: bytes) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
    """read_line's (meta_info, failure), and the HTTP status the failure names, if it names one."""
    try:
        chunk = json.loads(line[len(b"data:"):])
    except ValueError:
        return None, f"stream line that is not JSON: {line[:200]!r}", None
    if not isinstance(chunk, dict):
        return None, f"stream line that is not an object: {line[:200]!r}", None
    if "error" in chunk:
        error = chunk["error"]
        return None, f"stream error: {error}", _status(error.get("code") if isinstance(error, dict) else None)
    meta = chunk.get("meta_info") or {}
    finish = meta.get("finish_reason")
    if isinstance(finish, dict) and finish.get("type") == "abort":
        return None, f"aborted by the server: {finish.get('message')}", _status(finish.get("status_code"))
    return meta, None, None


def _status(value) -> Optional[int]:
    """An HTTP status, or None for a value that is not one (OpenAI's error ``code`` may also be a string)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600 else None


def read_line(line: bytes) -> Tuple[Optional[dict], Optional[str]]:
    """One ``data:`` line of a server's stream: (its meta_info, None) for an output chunk, or (None, the failure it
    reports): SGLang's error envelope, an output the server finished as aborted, or a line that is no JSON object. The
    line is parsed, not searched, since an answer's own text can make the bytes of a key: a chunk whose text is the
    token ``error``, or ends in a quote and ``error``, serializes as ``"error"``."""
    meta, error, _ = _parse(line)
    return meta, error


def stream_error(line: bytes) -> Optional[str]:
    """The failure one ``data:`` line of a server's stream reports, or None for an output chunk (read_line)."""
    return _parse(line)[1]


def stream_failure(line: bytes) -> Tuple[Optional[str], Optional[int]]:
    """(the failure one ``data:`` line reports, the HTTP status it names), or (None, None) for an output chunk. SGLang
    names one in its error envelope's ``code`` (400 for a request it refuses) and in an abort's ``status_code`` (500 for
    a KV transfer that failed); a line that is no JSON names none."""
    return _parse(line)[1:]

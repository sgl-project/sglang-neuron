# SPDX-License-Identifier: Apache-2.0
"""``POST /pull_weights``: bring this host's local checkpoint up to a published weight version (local_checkpoint.pull),
for a trainer that publishes weight deltas and then has the engine reload the local checkpoint through
``update_weights_from_disk``. Every rank of a Neuron server runs on one host, so the HTTP server pulls once for it."""

import asyncio
from http import HTTPStatus

from pydantic import BaseModel


class PullWeightsReqInput(BaseModel):
    # host-local checkpoint dir the pulled weights land in; seeded from the server's model path
    local_checkpoint_dir: str
    # shared dir the publisher writes weight_v{N:06d}/ version dirs under
    source_dir: str
    target_version: int


async def pull_weights(obj: PullWeightsReqInput):
    from fastapi.responses import ORJSONResponse
    from sglang.srt.entrypoints import http_server

    from sglang_neuron.weight_sync import local_checkpoint

    server_args = http_server._global_state.tokenizer_manager.server_args
    try:
        await asyncio.to_thread(local_checkpoint.pull, local_checkpoint_dir=obj.local_checkpoint_dir,
                                base_dir=server_args.model_path, source_dir=obj.source_dir,
                                target_version=obj.target_version)
        success, message = True, "Success."
    except Exception as e:
        success, message = False, f"pull_weights failed: {e!r}"
    return ORJSONResponse({"success": success, "message": message},
                          status_code=HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST)


def add_route(original_fn, *args, **kwargs):
    """AROUND http_server.launch_server: serve /pull_weights beside SGLang's own routes."""
    from sglang.srt.entrypoints.http_server import app

    if not any(getattr(route, "path", None) == "/pull_weights" for route in app.routes):
        app.add_api_route("/pull_weights", pull_weights, methods=["POST"])
    return original_fn(*args, **kwargs)

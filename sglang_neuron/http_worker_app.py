# SPDX-License-Identifier: Apache-2.0
"""The HTTP app a tokenizer worker of ``--tokenizer-worker-num`` > 1 serves: SGLang's own
(``sglang.srt.entrypoints.http_server:app``), imported after the worker loads the plugins.

A tokenizer worker is a fresh interpreter that imports the app by its path (``SGLANG_HTTP_SERVER_APP_PATH``) and builds
its TokenizerManager in the app's lifespan. Upstream SGLang's worker never calls ``load_plugins()``, so without this
app the plugin's tokenizer-side hooks (admission refusals, the abort logprob converter) would not exist in the workers
that serve the requests. ``sglang_patches/tokenizer_worker_plugins`` points the workers here."""

from sglang.srt.plugins import load_plugins

load_plugins()

from sglang.srt.entrypoints.http_server import app  # noqa: E402,F401

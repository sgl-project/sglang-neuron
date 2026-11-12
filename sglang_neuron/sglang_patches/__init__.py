# SPDX-License-Identifier: Apache-2.0
"""Changes to upstream SGLang that neither a platform method nor a subclass can make. Each is a candidate to upstream to
SGLang; each module's "Upstream:" line says what SGLang would need for the module to go away.

Most modules wrap or replace named upstream symbols through SGLang's HookRegistry, from a ``register`` function that
general_plugin calls (the PD modules are registered by disaggregation). tokenizer_worker_plugins has no ``register``:
the platform calls its ``configure`` at launch.

Before it changes anything, a module checks the source of every upstream symbol it wraps, replaces or relies on against
the SGLang commit it was written for (``source_guard``). If any source differs, the module raises instead, because a
hook on changed code could serve a path nobody has validated."""

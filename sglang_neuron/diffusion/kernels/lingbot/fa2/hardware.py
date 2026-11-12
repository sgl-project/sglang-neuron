# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The FlashAttention-2 primitive's per-generation engine plan (trace-time, pure Python).

:class:`~.flash_attention_fwd.FlashAttentionFwd` is one primitive for Trainium 2 and Trainium 3.
The generations differ only in where the two reduce chains and the exponential run, because two
instructions exist only on Trainium 3: ``activate2`` (a Scalar
eviction with a chained max) and ``exponential`` (a Vector exp that subtracts a positive max and
chains the row sum).

=================  ================================  ===========================================
stage              Trainium 2 (NeuronCore-v3)        Trainium 3 (NeuronCore-v4)
=================  ================================  ===========================================
MM1 eviction       Vector ``tensor_scalar_reduce``   Scalar ``activate2`` (the first point's fill
                   (scale, chained max)              on Vector ``tensor_scalar_reduce``)
max update         Vector max; Scalar rescale and    Vector max; Scalar rescale
                   the negated max for the exp
exponential        Scalar ``activation`` (bias =     Vector ``exponential`` (positive max)
                   -max), chained row sum            chained row sum
row-sum merge      GpSimd, two ``tensor_tensor``     GpSimd, one ``tensor_scalar``
drain P^T evict    Vector                            Scalar
=================  ================================  ===========================================

Everything else -- the geometry, the buffers and their placement, MM1/MM2, the key-bias matmul,
the P-transpose, the carried state and its merge into the output -- is shared. The max fold runs on
Vector on both generations: GpSimd has no float max datapath.

The kernel calls :func:`generation` with the NeuronCore version it is traced for; this module holds
no NKI so the plan is unit-testable on a CPU host.
"""

TRN2 = "trn2"
TRN3 = "trn3"

#: The trace-time requirement of the Trainium 3 plan.
TRN3_INSTRUCTIONS_REQUIRED = (
    "this NeuronCore lacks nisa.exponential and nisa.activate2, which the FA2 primitive's "
    "Trainium 3 plan uses"
)


def generation(gen4_or_later):
    """The engine plan for a NeuronCore of version >= v4 (Trainium 3) or v3 (Trainium 2)."""
    if gen4_or_later:
        return TRN3
    return TRN2


def has_required_instructions(plan, has_trn3_instructions):
    """Whether a NeuronCore can run ``plan``: the Trainium 3 plan needs ``nisa.exponential`` and
    ``nisa.activate2``; the Trainium 2 plan runs on every v3+ core."""
    if plan == TRN3:
        return has_trn3_instructions
    return True


def qk_evict_engine(plan, step, sk_tiles_per_blk):
    """``"vector"`` or ``"scalar"``: the engine of fill step ``step``'s MM1 eviction.

    Trainium 3 evicts the first point's fill on Vector (nothing else runs yet, so the two engines
    share the fill) and the rest on Scalar (``activate2``); Trainium 2 evicts on Vector throughout.
    """
    if plan == TRN3 and step >= sk_tiles_per_blk:
        return "scalar"
    return "vector"


def body_qk_evict_engine(plan):
    """The engine of the steady-state MM1 eviction."""
    if plan == TRN3:
        return "scalar"
    return "vector"


def drain_transpose_evict_engine(plan):
    """The engine of the drain's P^T evictions: the one not holding the row-sum chain."""
    if plan == TRN3:
        return "scalar"
    return "vector"


def exp_engine(plan):
    """The engine of the exponential and its chained row sum."""
    if plan == TRN3:
        return "vector"
    return "scalar"


def needs_negated_max(plan):
    """Whether the exponential takes the max negated (a Scalar ``activation`` bias)."""
    return plan == TRN2


def supports_split_tp_evict(plan):
    """``split_tp_evict`` (alternate P^T evictions onto Scalar) is a Trainium 2 tuning."""
    return plan == TRN2


def engine_plan(plan, sk_tiles_per_blk, qk_offset):
    """The whole plan as plain values (for tests and reports): the fill's MM1 eviction engines,
    then the steady-state and drain choices."""
    fill = []
    for step in range(qk_offset):
        fill.append(qk_evict_engine(plan, step, sk_tiles_per_blk))
    return {
        "fill_qk_evict": tuple(fill),
        "body_qk_evict": body_qk_evict_engine(plan),
        "exp": exp_engine(plan),
        "negated_max": needs_negated_max(plan),
        "drain_transpose_evict": drain_transpose_evict_engine(plan),
        "split_tp_evict": supports_split_tp_evict(plan),
    }

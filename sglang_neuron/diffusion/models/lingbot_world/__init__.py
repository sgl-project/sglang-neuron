# SPDX-License-Identifier: Apache-2.0
"""Neuron LingBot-World-V2 (causal-fast) components: the DiT, its paged cache attention and
runtime, and the real-time serving pieces built on them.

No eager imports: spawned worker processes import single modules of this package before they
pin their NeuronCores.
"""

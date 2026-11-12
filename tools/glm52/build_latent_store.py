#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Link a GLM-5.2 GK weight store for another latent split (``latent_tp``) of a decode point, without a rebuild.

The latent split shards only the down-projections ``wq_a_all`` / ``wkv_a_all``. A token-group-shard store
(``latent_tp`` 0, such as the 128k point's) holds them at the token group's width, and the loader slices a latent
rank's columns at load. So a split's store is the point's store with those two files of every rank linked from the
token-group-shard store, and a ``geometry.json`` that records the split. Every other rank file links to the point's
store. The token-group-shard store must place the down-projections as the point's store does, converted from the same
checkpoint (``LATENT_PLACEMENT_KEYS``).

Usage:
    build_latent_store.py <point store> <token-group-shard store> <new store> <latent_tp>

Example (the 1M point at latent_tp 8)::

    build_latent_store.py <stores>/1m <stores>/128k <stores>/1m_lat8 8

Links point at absolute paths, so store paths may be relative. The tool writes the new ``geometry.json``, copies the
point store's other top-level files, links its other top-level directories (such as a shared-expert overlay, which
no split changes), and never touches an existing path.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

LATENT_FILES = ("wq_a_all.npy", "wkv_a_all.npy")
#: The ``geometry.json`` fields that decide which columns and layers a rank's ``wq_a_all`` / ``wkv_a_all`` hold and
#: which checkpoint they come from. The loader reads only the new store's ``geometry.json`` (the point store's), so a
#: token store that differs in any of them would load the wrong weights at the right shapes.
LATENT_PLACEMENT_KEYS = ("ranks", "dp_size", "cp_degree", "head_shard", "num_layers", "layer_offset",
                         "checkpoint_format", "checkpoint_digest")


def build(point_store: str, token_store: str, new_store: str, latent_tp: int) -> int:
    """Link ``new_store`` as described in the module docstring and return the number of links made.

    Raises:
        FileExistsError: If ``new_store`` exists.
        ValueError: If ``token_store`` is not a token-group-shard store, holds the down-projections otherwise than
            ``point_store`` (``LATENT_PLACEMENT_KEYS``), or a rank lacks a latent file there.
    """
    # a link holds its target as written, and a relative target resolves against the link's own directory
    point_store, token_store = os.path.abspath(point_store), os.path.abspath(token_store)
    if os.path.exists(new_store):
        raise FileExistsError(f"{new_store} exists; this tool never touches an existing path")
    with open(os.path.join(point_store, "geometry.json")) as f:
        point_meta = json.load(f)
    with open(os.path.join(token_store, "geometry.json")) as f:
        token_meta = json.load(f)
    if int(token_meta.get("latent_tp", 0)) != 0:
        raise ValueError(f"{token_store} was built for latent_tp={token_meta['latent_tp']}: the latent files must be "
                         "the token-group shard (latent_tp 0)")
    differ = [f"{key}={token_meta.get(key)} against the point's {point_meta.get(key)}" for key in LATENT_PLACEMENT_KEYS
              if point_meta.get(key) != token_meta.get(key)]
    if differ:
        raise ValueError(f"{token_store} holds wq_a / wkv_a otherwise than {point_store} ({', '.join(differ)}): link "
                         "the latent files from a token-group-shard store of the point's sharding and checkpoint")
    ranks = sorted(d for d in os.listdir(point_store) if d.startswith("rank"))
    for r in ranks:
        for f in LATENT_FILES:
            if not os.path.exists(os.path.join(token_store, r, f)):
                raise ValueError(f"{token_store}/{r} has no {f}")
    os.makedirs(new_store)
    links = 0
    for f in os.listdir(point_store):
        if f.startswith("rank"):
            continue
        if f == "geometry.json":
            meta = dict(point_meta)
            meta["latent_tp"] = int(latent_tp)
            meta["latent_store"] = (f"wq_a_all / wkv_a_all of every rank are {token_store}'s token-group shards, "
                                    "sliced at load")
            with open(os.path.join(new_store, f), "w") as dst:
                json.dump(meta, dst, indent=1)
        elif os.path.isdir(os.path.join(point_store, f)):
            os.symlink(os.path.join(point_store, f), os.path.join(new_store, f))
            links += 1
        else:
            shutil.copy2(os.path.join(point_store, f), os.path.join(new_store, f))
    for r in ranks:
        os.makedirs(os.path.join(new_store, r))
        for f in os.listdir(os.path.join(point_store, r)):
            src_dir = token_store if f in LATENT_FILES else point_store
            os.symlink(os.path.join(src_dir, r, f), os.path.join(new_store, r, f))
            links += 1
    return links


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    print("links", build(argv[1], argv[2], argv[3], int(argv[4])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

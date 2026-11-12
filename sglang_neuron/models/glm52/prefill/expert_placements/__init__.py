# SPDX-License-Identifier: Apache-2.0
"""The static MoE expert placement the PP4 prefill point serves, pinned by md5.

``place_pp4_u9.json`` gives, for each MoE layer 3-77, the expert of each of the 288 slots of a PP4 stage's EP32 group
(``e_local`` 9 a rank, -1 an empty slot), so each stage's group places only the layers it holds. It was calibrated on
the routing of an agentic workload (its ``calibration`` field) so that a hot expert's rank holds fewer experts;
tools/glm52/build_placement.py builds a placement of this form from another routing histogram. The recipe sets
``GLM_GK_MOE_PLACEMENT`` to it, which the prefill loaders read (layout/prefill/weight_loaders.moe_placement) both when the
MoE modules are built (the slots per rank) and when the router, the bias and the expert list load in slot order. Another
placement routes the same tokens to other slots, so the experts per rank, the per-rank expert blocks and the device time
differ from the measured point: the md5 pin makes a stale or edited copy refuse to load instead.

Standard library only, so the launching process and the catalog recipe can name the file without importing torch.
"""

import hashlib
import os

#: The served placement file.
PLACEMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "place_pp4_u9.json")
#: md5 of place_pp4_u9.json, whose slot table is the one the PP4 stages were measured with.
PLACEMENT_MD5 = "340419ffa2b4122fb40f14046de4c889"
#: The variable the prefill loaders read the placement path from (layout/prefill/weight_loaders.MOE_PLACEMENT_ENV).
PLACEMENT_ENV = "GLM_GK_MOE_PLACEMENT"


def file_md5(path: str) -> str:
    """The md5 hex digest of a file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_placement(pp_size: int, environ=os.environ) -> str:
    """The placement file ``GLM_GK_MOE_PLACEMENT`` names, refused unless its md5 is ``PLACEMENT_MD5``.

    Unset is refused too: the prefill loaders then build the identity placement (8 slots a rank, no empty slot),
    which loads and runs, but is not the program the PP4 stages were measured on. That holds for the PP4 stages, which
    the file places (EP32, 9 slots a rank); the PP16 stages (EP8, recipe.point_settings) run the identity placement (32
    experts a rank, which the file's EP32 slots do not fit), so there the variable must be unset (returns "").
    """
    path = environ.get(PLACEMENT_ENV, "").strip()
    if int(pp_size) == 16:
        if path:
            raise ValueError(f"{PLACEMENT_ENV}={path}: the PP16 prefill's 8-rank stages run the identity expert "
                             "placement (32 experts a rank); the PP4 stages' placement does not fit them, so unset it")
        return ""
    if not path:
        raise ValueError(f"{PLACEMENT_ENV} is unset; the GLM-5.2 prefill serves the static expert placement "
                         f"{PLACEMENT_FILE} (the catalog recipe sets it).")
    try:
        found = file_md5(path)
    except OSError as e:
        raise ValueError(f"{PLACEMENT_ENV}={path} cannot be read: {e}.") from None
    if found != PLACEMENT_MD5:
        raise ValueError(f"{PLACEMENT_ENV}={path} has md5 {found}, and the served placement is {PLACEMENT_FILE} (md5 "
                         f"{PLACEMENT_MD5}); another placement places the experts in other slots.")
    return path

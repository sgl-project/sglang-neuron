# SPDX-License-Identifier: Apache-2.0
"""
Hardware Configuration for Trainium Instances

This module provides hardware-specific configurations for different Trainium instance types,
including EFA (Elastic Fabric Adapter) BDF mappings for optimal NUMA placement.

BDF (Bus:Device:Function) addresses are stable PCI identifiers that don't change,
while EFA interface names (e.g., rdmapXXXs0, efa_X) can vary between boots/driver versions.
"""

import logging
import os

logger = logging.getLogger(__name__)


# per instance family, the EFA BDF address of each Neuron device index
_EFA_BDFS: dict[str, tuple[str, ...]] = {
    "trn2": (
        "0000:c9:00.0",  # ND 0
        "0000:b4:00.0",  # ND 1
        "0000:b3:00.0",  # ND 2
        "0000:ca:00.0",  # ND 3
        "0000:6c:00.0",  # ND 4
        "0000:57:00.0",  # ND 5
        "0000:56:00.0",  # ND 6
        "0000:6d:00.0",  # ND 7
        "0000:98:00.0",  # ND 8
        "0000:83:00.0",  # ND 9
        "0000:82:00.0",  # ND 10
        "0000:99:00.0",  # ND 11
        "0000:f5:00.0",  # ND 12
        "0000:e0:00.0",  # ND 13
        "0000:df:00.0",  # ND 14
        "0000:f6:00.0",  # ND 15
    ),
    "trn3pds": (
        "0000:a9:00.0",  # ND 0
        "0000:a8:00.0",  # ND 1
        "0000:bf:00.0",  # ND 2
        "0000:be:00.0",  # ND 3
        "0000:d5:00.0",  # ND 4
        "0000:d4:00.0",  # ND 5
        "0000:eb:00.0",  # ND 6
        "0000:ea:00.0",  # ND 7
        "0000:4c:00.0",  # ND 8
        "0000:4b:00.0",  # ND 9
        "0000:62:00.0",  # ND 10
        "0000:61:00.0",  # ND 11
        "0000:78:00.0",  # ND 12
        "0000:77:00.0",  # ND 13
        "0000:8e:00.0",  # ND 14
        "0000:8d:00.0",  # ND 15
    ),
}


def get_instance_family() -> str:
    """The instance family, from the DMI product name (``/sys/devices/virtual/dmi/id/product_name``) by prefix:
    ``trn3*`` is ``"trn3pds"``, ``trn2*`` is ``"trn2"``, and any other name's part before its first dot is returned as
    it is; an unreadable product name raises ``RuntimeError``."""
    sysfs_path = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(sysfs_path) as f:
            product_name = f.read().strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot read instance type from {sysfs_path}") from e

    # Extract family from "trn2.48xlarge" -> "trn2", then default by prefix.
    family = product_name.split(".")[0]
    if family.startswith("trn3"):
        return "trn3pds"
    if family.startswith("trn2"):
        return "trn2"
    return family


def get_efa_bdf_mapping() -> tuple[str, ...]:
    """The EFA PCI addresses (BDF) of this instance family, indexed by Neuron device number; an unknown or unsupported
    family raises."""
    family = get_instance_family()

    if family not in _EFA_BDFS:
        supported = ", ".join(sorted(_EFA_BDFS.keys()))
        raise ValueError(
            f"Unsupported instance family: {family}. Supported: {supported}"
        )

    return _EFA_BDFS[family]


def get_efa_interface_from_bdf(bdf: str) -> str:
    """The EFA interface name (``rdmap201s0``) of PCI address ``bdf`` (``0000:c9:00.0``) from sysfs; none found raises
    ``RuntimeError``."""
    infiniband_path = f"/sys/bus/pci/devices/{bdf}/infiniband"
    try:
        interfaces = os.listdir(infiniband_path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"No EFA device found at {infiniband_path} for BDF {bdf}. "
            f"This is expected on instances without EFA (e.g. trn3 3xlarge). "
            f"EFA affinity is a CPU performance optimization, not a "
            f"correctness requirement. If you are using an instance that "
            f"supports EFA, we recommend installing EFA to improve performance."
        ) from e

    if not interfaces:
        raise RuntimeError(
            f"No infiniband interfaces found in {infiniband_path} for BDF {bdf}. "
            f"The EFA device may not be properly initialized."
        )

    return interfaces[0]


# The physical NeuronCores grouped into one logical core when NEURON_LOGICAL_NC_CONFIG is unset; the platform sets it
# for every served launch (platform.set_logical_nc_config).
_DEFAULT_LNC_CONFIG = 2
_SUPPORTED_LNC_CONFIGS = (1, 2)


def get_logical_nc_config() -> int:
    """The logical NeuronCore degree, 1 or 2: ``NEURON_LOGICAL_NC_CONFIG``, the value the runtime reads to fuse physical
    cores into a logical one, default 2. The compiler must get the same value (``--logical-nc-config``) so a NEFF's core
    grouping matches the runtime's. A served launch runs 2, or 1 on the giga-kernel decode: the platform sets the
    variable for its schedulers (platform.set_logical_nc_config). NKI launch grids read it too, so another degree raises
    ``ValueError`` here rather than inside the compiler."""
    raw = os.environ.get("NEURON_LOGICAL_NC_CONFIG", str(_DEFAULT_LNC_CONFIG))
    try:
        lnc_config = int(raw)
    except ValueError:
        lnc_config = None
    if lnc_config not in _SUPPORTED_LNC_CONFIGS:
        raise ValueError(
            "Only LNC-1 and LNC-2 are supported; got "
            f"NEURON_LOGICAL_NC_CONFIG={raw!r}. Set the variable to 1 or 2, "
            "or leave it unset for the default of 2."
        )
    return lnc_config


def get_efa_interface(local_rank: int, visible_devices: list[int]) -> str:
    """The EFA interface closest to local rank ``local_rank``'s core.

    Walks core -> Neuron device -> host index (the device node's minor number) -> EFA PCI address -> interface, with
    the core taken from ``visible_devices`` (parsed ``NEURON_VISIBLE_DEVICES``). Raises ``RuntimeError`` if the device
    node can't be stat'ed or the interface isn't found, and ``IndexError`` for a device past the instance's EFA
    table."""
    lnc = visible_devices[local_rank]

    lnc_config = get_logical_nc_config()
    nd = lnc * lnc_config // 8

    try:
        device_stat = os.stat(f"/dev/neuron{nd}")
    except OSError as e:
        raise RuntimeError(
            f"Cannot access /dev/neuron{nd}: {e}. "
            f"Verify device exists and permissions are correct. "
            f"local_rank={local_rank}, visible_devices={visible_devices}"
        ) from e

    host_nd = os.minor(device_stat.st_rdev)
    efa_bdf_mapping = get_efa_bdf_mapping()

    if host_nd >= len(efa_bdf_mapping):
        raise IndexError(
            f"Computed Neuron device index {host_nd} exceeds available EFA BDF "
            f"mappings (max index: {len(efa_bdf_mapping) - 1}). "
            f"local_rank={local_rank}, visible_devices={visible_devices}, "
            f"lnc_config={lnc_config}"
        )

    bdf = efa_bdf_mapping[host_nd]
    return get_efa_interface_from_bdf(bdf)


def parse_range_list(range_list_str: str) -> list[int]:
    """The integers of a range list such as ``NEURON_VISIBLE_DEVICES`` or a sysfs CPU list: ``"0-2,5,7-9"`` is ``[0, 1,
    2, 5, 7, 8, 9]``."""
    result = []
    for start_end in range_list_str.split(","):
        try:
            start, end = start_end.split("-")
        except ValueError:
            start = start_end
            end = start_end
        result.extend(range(int(start), int(end) + 1))
    return result

# SPDX-License-Identifier: Apache-2.0
"""Fetch named tensors of a Hugging Face safetensors checkpoint by HTTP range reads into one local safetensors file,
without downloading the shards (a MiMo-V2.6 layer's fused QKV is 60 MB of a 5 GB shard).

    python3 tools/mimo_v2/fetch_tensors.py --repo XiaomiMiMo/MiMo-V2.6-Flash-RL --out /tmp/mimo_slices \\
        model.layers.0.self_attn.qkv_proj.weight model.layers.0.self_attn.qkv_proj.weight_scale_inv ...

The output directory gets ``model.safetensors`` with the tensors as stored (dtypes and shapes unchanged), the
checkpoint's ``config.json`` and its ``model.safetensors.index.json`` metadata, so tools/mimo_v2/check_layout.py reads
it as a checkpoint."""

import argparse
import json
import os
import struct
import urllib.request

_BASE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"


def _get(url: str, start: int = None, end: int = None, token: str = None) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.out, exist_ok=True)
    url = lambda name: _BASE.format(repo=args.repo, revision=args.revision, name=name)  # noqa: E731
    index = json.loads(_get(url("model.safetensors.index.json"), token=token))
    with open(os.path.join(args.out, "config.json"), "wb") as f:
        f.write(_get(url("config.json"), token=token))
    by_file = {}
    for name in args.names:
        by_file.setdefault(index["weight_map"][name], []).append(name)
    header, blobs, offset = {}, [], 0
    for file, names in sorted(by_file.items()):
        n = struct.unpack("<Q", _get(url(file), 0, 7, token))[0]
        meta = json.loads(_get(url(file), 8, 8 + n - 1, token))
        for name in names:
            start, end = meta[name]["data_offsets"]
            data = _get(url(file), 8 + n + start, 8 + n + end - 1, token)
            header[name] = {"dtype": meta[name]["dtype"], "shape": meta[name]["shape"],
                            "data_offsets": [offset, offset + len(data)]}
            blobs.append(data)
            offset += len(data)
            print(f"{name}: {meta[name]['dtype']} {meta[name]['shape']} ({len(data) / 2**20:.1f} MiB)")
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(os.path.join(args.out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)
    weight_map = {name: "model.safetensors" for name in header}
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index.get("metadata") or {}, "weight_map": weight_map}, f, indent=1)


if __name__ == "__main__":
    main()

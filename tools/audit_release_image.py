#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The checks of tools/audit_release_image.sh on a release image and the base image it was built on. Standard library
only: ``versions`` runs inside the image on its own interpreter.

``scan`` reads ``docker save <base> <image>`` on stdin in one pass, nothing extracted. Every path of every layer is
checked, and every byte of the image's own layers read (a shared base layer's bytes with ``--base-content`` only), the
files a later layer deletes or replaces included: ``docker save`` (and a registry) hands out each layer whole, and a
whiteout hides a file from the container only. A finding is

- FAIL in the image's own layers. With ``--flattened`` (an image that holds the base's files in its own layer, not
  one built FROM the base), an own-layer file with the same path, type and size as in the base's final file system is
  the base's (inherited). Without it, an own-layer file at a base path is a rewrite and is held to every rule. A
  directory holds no bytes: an own layer lists a base directory again when a build step touches it (a mount point
  under /tmp), and with the base's mode, owner and extended attributes it is the base's directory either way;
- BASE in a layer the image shares with the base, or for an inherited file: reported, since the base is published as
  it is;
- REPORT for an internal-name pattern in a third-party file the image adds (SGLang, its dependencies): read, not
  failed.

The rules:

- forbid: anything at or under a ``--forbid`` path, in any layer, the base's included (the QA checkout the public
  image strips);
- restricted: anything under a ``--restrict`` directory, in any layer, that is not a ``--restrict-allow`` path, under
  one, or a directory above one (the public image's /opt/sglang-neuron holds exactly the allowed copies);
- whiteout: a file of an own layer that a later layer deletes (its bytes stay in the layer that added it);
- source: a ``.git`` directory; a project root (``pyproject.toml``, ``setup.py`` or ``setup.cfg`` outside the
  installed packages and the system's trees and outside an ``--allow-source`` tree) and its ``tests``, ``ci``,
  ``docs``, ``scripts``; a ``sglang_neuron`` package outside site-packages and the ``--allow-source`` trees;
- cache: a compile cache directory, a NEFF or a Neuron profile;
- key: an SSH or TLS private key (by name, or a PEM private-key block in its bytes), ``.netrc``, AWS, Docker, git or
  PyPI credentials;
- token: a Hugging Face cache or token, a Hugging Face or AWS access key in a file's bytes (AWS's documented
  ``...EXAMPLE`` keys pass), a credential-like variable or a token in the image config's environment or history. A
  key or a token in the bytes of a file a published package installs (pip RECORD; not the plugin) is that package's
  test vector: REPORT, when the file's bytes are the ones the RECORD hashes (a build step that writes over the
  package's file fails); a file the RECORD lists with no hash (the bytecode pip compiles at install) or a module's
  bytecode it does not list is the package's by its path;
- elf: a shared object or executable an own layer adds that no installed package owns (pip RECORD, dpkg list), or
  one in the plugin or an ``--allow-source`` tree (one in SGLang's own wheel, its GPU code objects, is REPORT);
- names: any pattern of ``--names FILE`` (the release process's own hosts, aliases, paths and account ids, one
  regular expression a line: the repository holds none) in a text file (no ELF header and no NUL byte in its first
  8 KiB, whatever its name) of an own layer, a version a later layer replaces included, that is our content: the
  sglang-neuron distribution's files, a file no package owns (the NIXL notice, the QA checkout) (FAIL). The same in a
  file another published package installs (pip RECORD, dpkg list), such as an author's name in its metadata, is
  REPORT with the owner named, when the file's bytes are the package's: the sha256 its RECORD names, the md5 of the
  deb's md5sums (a file either records no hash for, such as a deb's conffile, which dpkg sums in its status file
  instead, is the package's by its path). A private address, a user's home path or an account id, which
  documentation examples also hold, is REPORT, for review;
- leftover: an own layer's file matching a ``--leftover`` regular expression (build leftovers: wheels, /tmp, caches);
- own-only: an own layer's entry outside every ``--own-only`` path and its parent directories (the QA image adds its
  checkout to the public image and nothing else);
- replace (with ``--adds-only``): an own layer that changes or deletes an entry of a layer below it, of any kind: a
  file's bytes, a link's target, an entry's kind (a file made a directory or a link), a directory's mode, owner or
  extended attributes (a QA image built FROM the public image must ship the public image's files unchanged; a
  directory listed again with its mode, owner and extended attributes is no change);
- require: a ``--require`` path missing from the image's final file system;
- prefix (with ``--layer-prefix``): the base's layers are not the image's first layers, in order (the image is built
  FROM the base, so every layer the base was tested with ships unchanged);
- config (with ``--same-config``): the image's runtime config (environment, entrypoint, command, user, ports,
  healthcheck, ...) or a label (but ``--same-config-except`` ones) differs from the base's;
- nixl (with ``--nixl-patches DIR``, the checkout's patches/nixl): NIXL's distributions (``nixl``, ``nixl_cu13``)
  installed, each dist-info with its Apache-2.0 ``licenses/LICENSE``; the notice directory (``--nixl-notice``) with a
  README.md or NOTICE.md naming each ``--nixl-ids`` commit and every patch of DIR, byte for byte, and no other
  patch.

Outputs in ``--out``: report.json (every finding and count; with ``--hash-under``, the sha256 of each file under those
paths in the final file system), findings.txt, own_layer_paths.txt (each own layer's
entries: layer index, path), the final file lists of both images
(path, type, size, mode, uid:gid, link, xattrs, layer: ``release_files.txt`` and ``base_files.txt``, comparable line
by line), what the image adds and changes over the base (``added_files.txt``, ``changed_files.txt``,
``added_outside_home.txt``), the layers with their xattr counts (layers.json), and where the plugin is installed.

``versions`` prints, as JSON, the image's Python, OS, Neuron SDK packages (pip and apt), torch, torch-neuronx, SGLang
(its version from its source tree, and the commit pip recorded where it is installed), sglang-neuron, NIXL and
transformers.

Exit status 0 with no FAIL, 1 with one, 2 on a usage or input error."""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import subprocess
import sys
import tarfile

# a file up to this long is read whole; a longer one is hashed and searched in windows of this size, each starting
# SCAN_OVERLAP bytes before the last one ended, so a key, token or name across a window's edge is found too
CONTENT_LIMIT = 1 << 20
SCAN_OVERLAP = 64 << 10
# the package file lists (pip RECORD, dpkg .list) are read up to this size: SGLang's RECORD is over 1 MB
KEEP_LIMIT = 64 << 20
# tarfile's stream reads (and skips what it does not read) in blocks of this size. Larger is not faster: each small
# read (a 512-byte header, a small file) copies the rest of the buffer (_Stream.__read's `self.buf = t[size:]`), so
# 1 MiB read a 20,000-entry layer of the public image's size mix more slowly than 64 KiB (outputs identical); 64 KiB
# also takes a sixteenth of the reads of tarfile's default 10 KiB
TAR_BUFSIZE = 64 << 10
# directories that hold installed packages or the system's own files: a project file there is a package's
INSTALLED_TREES = re.compile(r"(^|/)(site-packages|dist-packages)/|^usr/share/|^usr/(local/)?lib/python3[^/]*/"
                             r"|^opt/conda/pkgs/|^usr/src/linux-headers")
REPO_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")
REPO_SUBDIRS = ("tests", "ci", "docs", "scripts")
CACHE_DIRS = re.compile(r"(^|/)(neuron-compile-cache|neuron_compile_cache|neff_cache|hlo_cache|nki_cache|"
                        r"neuroncc_compile_workdir|neuronxcc-[0-9][^/]*)(/|$)")
CACHE_FILES = re.compile(r"\.(neff|ntff)$|\.ntrace\.pb$")
KEY_NAMES = re.compile(r"(^|/)(id_rsa|id_dsa|id_ecdsa|id_ed25519)$|\.(key|p12|pfx|jks|keystore)$|(^|/)\.netrc$|"
                       r"(^|/)\.git-credentials$|(^|/)\.pypirc$|(^|/)\.aws/(credentials|config)$|"
                       r"(^|/)\.docker/config\.json$|(^|/)\.ssh/(?!known_hosts$)[^/]+$")
HF_PATHS = re.compile(r"(^|/)\.cache/huggingface/|(^|/)\.huggingface/|(^|/)huggingface/(token|stored_tokens)$")
# a PEM private-key block: the header and a body (a program that only names the header, like ssh-keygen, passes)
PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{64}")
TOKENS = {"a Hugging Face token": re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34}(?![A-Za-z0-9])"),
          "an AWS access key": re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)(?![A-Z0-9]{9}EXAMPLE)[A-Z0-9]{16}"
                                          rb"(?![A-Z0-9])")}
# a credential's variable name: its last _-separated part (HF_TOKEN, OPENAI_API_KEY, AWS_SECRET_ACCESS_KEY,
# DB_PASSWORD), not any name that holds the word (TOKENIZERS_PARALLELISM, SGLANG_MAX_TOKENS); a switch's value is none
SECRET_ENV = re.compile(r"(?:^|_)(?:TOKEN|SECRET(?:_KEY)?|PASSWORD|PASSWD|CREDENTIALS?|(?:API|ACCESS|PRIVATE)_KEY)$",
                        re.I)
NOT_A_SECRET = {"0", "1", "true", "false", "yes", "no", "on", "off", "none"}
ELF = b"\x7fELF"
WHITEOUT, OPAQUE = ".wh.", ".wh..wh..opq"
# the names rule's text files: no ELF header and no NUL byte in this many first bytes, by content, since dotfiles,
# logs and .pth files hold text under any name
TEXT_SNIFF = 8 << 10
# names that hold no organisation's own string: private addresses, user home paths, account ids. The release
# process gives its own names (hosts, aliases, cluster paths) in --names, so the repository names none.
GENERIC_NAMES = [
    ("a private address", re.compile(rb"(?<![\d.])10\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])|"
                                     rb"(?<![\d.])172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}(?![\d.])|"
                                     rb"(?<![\d.])192\.168\.\d{1,3}\.\d{1,3}(?![\d.])")),
    ("a user's home path", re.compile(rb"/(?:local/)?home/[a-z][\w-]{2,}/")),
    ("an account id", re.compile(rb"\b\d{12}\.dkr\.ecr\.|arn:aws[\w-]*:[\w-]*:[\w-]*:\d{12}:")),
]
APACHE = (b"Apache License", b"Version 2.0")


def norm(name: str) -> str:
    """A layer member's path, relative to the image root, without ``./`` and trailing ``/``."""
    name = posixpath.normpath(name.lstrip("/")) if name not in ("", ".", "./") else ""
    return "" if name == "." else name


def under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r + "/") for r in roots)


class _Prefixed(io.RawIOBase):
    """A read-only stream of ``head`` and then the rest of ``rest``."""

    def __init__(self, head: bytes, rest):
        self.head, self.rest = head, rest

    def readable(self):
        return True

    def readinto(self, buf):
        if self.head:
            n = min(len(buf), len(self.head))
            buf[:n], self.head = self.head[:n], self.head[n:]
            return n
        data = self.rest.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def _layer_kind(head: bytes) -> str | None:
    """The tar compression of a blob from its first bytes, or None for a blob that is no layer."""
    if head[:2] == b"\x1f\x8b":
        return "gz"
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "zst"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return ""
    if len(head) == 512 and not any(head):
        return ""  # an empty layer: the tar's end-of-archive blocks only
    return None


def read_names(path: str) -> list:
    """The patterns of a --names file, one regular expression a line (blank and # lines skipped), as compiled byte
    patterns. A pattern that does not compile is an error naming its line, and so is a file with no pattern at all, so
    that no gate records "names checked" without one; docker/build.sh reads the file with this before it builds."""
    patterns = []
    for number, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                patterns.append(re.compile(line.encode()))
            except re.error as error:
                raise ValueError(f"--names {path} line {number}: {line!r} is no regular expression ({error})") from None
    if not patterns:
        raise ValueError(f"--names {path}: no pattern (only blank or # lines)")
    return patterns


class Rules:
    """What a scan looks for, from the command line."""

    def __init__(self, args):
        self.forbid = [norm(p) for p in args.forbid]
        self.restrict = [norm(p) for p in args.restrict]
        self.restrict_allow = [norm(p) for p in args.restrict_allow]
        outside = [p for p in self.restrict_allow if not under(p, self.restrict)]
        if outside:
            raise ValueError(f"--restrict-allow {outside[0]} is under no --restrict directory")
        self.allow_source = [norm(p) for p in args.allow_source]
        self.nixl_notice = norm(args.nixl_notice) if args.nixl_patches else None
        self.nixl_patches = {}
        if args.nixl_patches:
            for name in sorted(os.listdir(args.nixl_patches)):
                if name.endswith(".patch"):
                    self.nixl_patches[name] = open(os.path.join(args.nixl_patches, name), "rb").read()
            if not self.nixl_patches:
                raise ValueError(f"--nixl-patches {args.nixl_patches}: no .patch file")
        self.nixl_ids = [i for i in (args.nixl_ids or "").split(",") if i]
        self.hash_under = [norm(p) for p in args.hash_under]
        self.names = list(GENERIC_NAMES)
        self.leftovers = [re.compile(p) for p in args.leftover]
        self.own_only = [norm(p) for p in args.own_only]
        self.extra = [("a --names pattern", rx) for rx in read_names(args.names)] if args.names else []
        self.content_limit = args.content_limit
        self.base_content = args.base_content


class Layer:
    """One layer: path -> entry fields, its whiteouts and opaque directories, its findings (rule, path, detail), the
    bytes the end of the scan needs (RECORD and dpkg lists, the NIXL overlay's notices), its ELF files, its name hits,
    and the hashes of each file with a key, token or name in it, as packages record them (_sums)."""

    def __init__(self, digest: str, own: bool):
        self.digest, self.own = digest, own
        self.entries, self.whiteouts, self.opaque, self.findings = {}, [], [], []
        self.kept, self.elf, self.names, self.hashes, self.sums = {}, [], [], {}, {}
        self.bytes = self.xattr_files = 0


def path_findings(path: str, kind: str, rules: Rules) -> list:
    """The rules a path breaks by its name alone: (rule, path, detail)."""
    out = []
    parts = path.split("/")
    if rules.forbid and under(path, rules.forbid):
        out.append(("forbid", path, "under a --forbid path"))
    if rules.restrict and under(path, rules.restrict) and not (
            under(path, rules.restrict_allow) or any(a.startswith(path + "/") for a in rules.restrict_allow)):
        out.append(("restricted", path, "under a --restrict directory and no --restrict-allow path"))
    if ".git" in parts:
        out.append(("source", path, "a .git directory"))
    if "sglang_neuron" in parts and not INSTALLED_TREES.search(path) and not under(path, rules.forbid +
                                                                                    rules.allow_source):
        out.append(("source", path, "a sglang_neuron tree outside site-packages"))
    if CACHE_DIRS.search(path) or (kind == "file" and CACHE_FILES.search(path)):
        out.append(("cache", path, "a compile cache or profile file"))
    if kind == "file" and KEY_NAMES.search(path):
        out.append(("key", path, "a key or credential file"))
    if HF_PATHS.search(path):
        out.append(("token", path, "a Hugging Face cache or token"))
    return out


# the rules own_findings applies (an own layer's only)
OWN_RULES = ("leftover", "own-only")


def own_findings(path: str, kind: str, rules: Rules) -> list:
    """The rules only the image's own layers are held to."""
    out = [("leftover", path, f"a build leftover ({rx.pattern})") for rx in rules.leftovers if rx.search(path)]
    if rules.own_only and not under(path, rules.own_only) and not any(r.startswith(path + "/")
                                                                       for r in rules.own_only):
        out.append(("own-only", path, "outside every --own-only path"))
    return out


def content_findings(path: str, data: bytes) -> list:
    out = []
    if PRIVATE_KEY.search(data):
        out.append(("key", path, "holds a PEM private key"))
    for what, rx in TOKENS.items():
        m = rx.search(data)
        if m:
            out.append(("token", path, f"holds {what} ({m.group(0)[:6].decode()}...)"))
    return out


def search(layer: "Layer", path: str, data: bytes, own: bool, text: bool, rules: Rules, seen: set) -> None:
    """Search ``data`` (a file, or one window of it) for keys and tokens, and, in a text file of an own layer, for
    names; ``seen`` keeps a file's findings to one each across its windows (one a key or token, one a pattern)."""
    for finding in content_findings(path, data):
        if finding not in seen:
            seen.add(finding)
            layer.findings.append(finding)
    if own and text:
        for what, rx in rules.names + rules.extra:
            m = rx.search(data)
            if m and rx not in seen:
                seen.add(rx)
                layer.names.append((path, what, m.group(0)[:80].decode(errors="replace"),
                                    any(rx is r for _, r in rules.extra)))


def _sums(sha, md5) -> tuple:
    """A file's hashes as its package records them: a pip RECORD's ``sha256=`` (urlsafe base64, no padding) and dpkg's
    md5sums' ``md5=`` (hex)."""
    return f"sha256={base64.urlsafe_b64encode(sha.digest()).rstrip(b'=').decode()}", f"md5={md5.hexdigest()}"


def _keep(path: str, rules: Rules) -> bool:
    """Whether the scan keeps this file's bytes for the end: pip RECORDs, dpkg file lists and md5sums, NIXL's licences
    and notice."""
    if path.endswith(".dist-info/RECORD") or (path.startswith("var/lib/dpkg/info/")
                                              and path.endswith((".list", ".md5sums"))):
        return True
    if re.search(r"(^|/)nixl(_cu\d+)?-[^/]+\.dist-info/licenses/", path):
        return True
    return bool(rules.nixl_notice and path.startswith(rules.nixl_notice + "/"))


def read_layer(digest: str, stream, kind: str, own: bool, rules: Rules) -> Layer:
    """One layer tar, read from ``stream`` once."""
    if kind == "zst":
        raise RuntimeError(f"{digest}: a zstd layer; save the images from a store that keeps layers as tar or gzip")
    layer = Layer(digest, own)
    with tarfile.open(fileobj=stream, mode="r|" + kind, bufsize=TAR_BUFSIZE) as tar:
        for member in tar:
            path = norm(member.name)
            if not path:
                continue
            base, parent = posixpath.basename(path), posixpath.dirname(path)
            if base == OPAQUE:
                layer.opaque.append(parent)
                continue
            if base.startswith(WHITEOUT):
                layer.whiteouts.append(posixpath.join(parent, base[len(WHITEOUT):]))
                continue
            kind_ = "dir" if member.isdir() else "link" if member.issym() or member.islnk() else \
                "file" if member.isreg() else "other"
            xattrs = sorted(k[len("SCHILY.xattr."):] for k in member.pax_headers if k.startswith("SCHILY.xattr."))
            layer.xattr_files += bool(xattrs)
            size = member.size if member.isreg() else 0
            layer.entries[path] = (kind_, size, member.mode, f"{member.uid}:{member.gid}", member.linkname or "",
                                   ",".join(xattrs))
            layer.bytes += size
            layer.findings += path_findings(path, kind_, rules)
            if own:
                layer.findings += own_findings(path, kind_, rules)
            deep = own or rules.base_content  # a shared base layer's bytes are the base's own scan
            keep = _keep(path, rules) and size <= KEEP_LIMIT
            hashed = bool(rules.hash_under) and under(path, rules.hash_under)
            if not member.isreg() or not (deep or keep or hashed):
                continue
            f = tar.extractfile(member)
            head = f.read(4) if f else b""
            if head == ELF:
                layer.elf.append(path)
            seen = set()
            if size > max(rules.content_limit, KEEP_LIMIT if keep else 0):
                if not (deep or hashed):
                    continue
                # hashed as it streams by: --hash-under's sha256, and, should a window hold a key, a token or a name,
                # the sums its package's pass is checked against (the bytes cannot be read again)
                sha = hashlib.sha256(head)
                md5 = hashlib.md5(head, usedforsecurity=False) if deep else None
                window = head
                text = None
                for chunk in iter(lambda: f.read(rules.content_limit), b""):
                    sha.update(chunk)
                    if deep:
                        md5.update(chunk)
                        window = window[-SCAN_OVERLAP:] + chunk if text is not None else window + chunk
                        if text is None:
                            text = head != ELF and b"\0" not in window[:TEXT_SNIFF]
                        search(layer, path, window, own, text, rules, seen)
                if hashed:
                    layer.hashes[path] = sha.hexdigest()
                if seen:
                    layer.sums[path] = _sums(sha, md5)
                continue
            data = head + (f.read() if f else b"")
            if hashed:
                layer.hashes[path] = hashlib.sha256(data).hexdigest()
            if keep:
                layer.kept[path] = data
            if deep:
                search(layer, path, data, own, head != ELF and b"\0" not in data[:TEXT_SNIFF], rules, seen)
                if seen:
                    layer.sums[path] = _sums(hashlib.sha256(data), hashlib.md5(data, usedforsecurity=False))
    return layer


def read_save(stream, base_layers: set, rules: Rules) -> tuple:
    """Every layer and every small JSON member (configs, manifests) of a ``docker save`` tar, read once:
    ({blob digest: Layer}, {member name: parsed JSON}). ``base_layers`` holds the base's layer digests."""
    layers, docs = {}, {}
    with tarfile.open(fileobj=stream, mode="r|", bufsize=TAR_BUFSIZE) as outer:
        for member in outer:
            if not member.isreg():
                continue
            f = outer.extractfile(member)
            head = f.read(512)
            kind = _layer_kind(head)
            name = norm(member.name)
            if kind is not None:
                digest = _blob_digest(name)
                layers[digest] = read_layer(digest, io.BufferedReader(_Prefixed(head, f), 1 << 20), kind,
                                            digest not in base_layers, rules)
            elif member.size <= 64 << 20:
                try:
                    docs[name] = json.loads(head + f.read())
                except ValueError:
                    pass
    return layers, docs


def _blob_digest(name: str) -> str:
    """The digest a member name stands for: blobs/sha256/<hex>, or a legacy <id>/layer.tar."""
    parts = name.split("/")
    return f"{parts[1]}:{parts[2]}" if len(parts) == 3 and parts[0] == "blobs" else name


def image_layers(docs: dict, image_id: str) -> tuple:
    """(the image's layer digests in order, its config) by the image id. Docker's own image store ids an image by its
    config's digest, which manifest.json names; the containerd image store by the digest of its OCI manifest or image
    index, a blob of the save, whose linux/amd64 manifest names the config and the layers."""
    want = image_id.split(":", 1)[-1]
    for entry in docs.get("manifest.json") or []:
        config = norm(entry["Config"])
        if config.split("/")[-1].removesuffix(".json") == want:
            return [_blob_digest(norm(p)) for p in entry["Layers"]], docs.get(config) or {}
    doc = docs.get(f"blobs/sha256/{want}")
    if isinstance(doc, dict) and isinstance(doc.get("manifests"), list):
        # an image index: its linux/amd64 image (an attestation manifest says unknown/unknown)
        images = [m for m in doc["manifests"] if (m.get("platform") or {}).get("os") == "linux"
                  and (m.get("platform") or {}).get("architecture") == "amd64"]
        if len(images) == 1:
            return image_layers(docs, images[0]["digest"])
        raise ValueError(f"image index {image_id} holds {len(images)} linux/amd64 images, not one")
    if isinstance(doc, dict) and isinstance(doc.get("config"), dict) and isinstance(doc.get("layers"), list):
        config = docs.get(f"blobs/sha256/{doc['config']['digest'].split(':', 1)[-1]}") or {}
        return [layer["digest"] for layer in doc["layers"]], config
    raise ValueError(f"image {image_id} is not in the saved tar (manifest.json, or an OCI manifest or index blob)")


def same_directory(old: tuple, new: tuple) -> bool:
    """Whether entry ``new`` is directory ``old`` listed again (entries are (kind, size, mode, owner, link, xattrs)): a
    build step that touches a directory (a mount point under /tmp, a file added in it) makes its layer list it again,
    and with the same mode, owner and extended attributes a container sees no change."""
    return old[0] == new[0] == "dir" and (old[2], old[3], old[5]) == (new[2], new[3], new[5])


def merged_view(layers: list) -> tuple:
    """The image's files as a container sees them, applying each layer's whiteouts in order: ({path: (entry, layer
    index)}, [(path, layer index, how, by layer index, kind)] for every entry a later layer deletes or replaces, of any
    kind (the lower entry's: file, dir, link, other), a directory listed again unchanged (same_directory) left out,
    {(layer index, whiteout path)} for each whiteout or opaque marker that hides an entry of a layer below). An opaque
    marker that docker puts on a directory its layer creates hides nothing."""
    view, shadowed, hiding = {}, [], set()
    children = collections.defaultdict(set)  # directory -> the paths in the view directly under it

    def drop(path: str, by: int) -> int:
        n = sum(drop(child, by) for child in list(children.pop(path, ())))
        if path in view:
            entry, index = view.pop(path)
            children[posixpath.dirname(path)].discard(path)
            n += 1
            shadowed.append((path, index, "deleted", by, entry[0]))
        return n

    for i, layer in enumerate(layers):
        for directory in layer.opaque:
            if sum(drop(child, i) for child in list(children.get(directory, ()))):
                hiding.add((i, directory))
        for path in layer.whiteouts:
            if drop(path, i):
                hiding.add((i, path))
        for path, entry in layer.entries.items():
            old = view.get(path)
            if old is not None and not same_directory(old[0], entry):
                if old[0][0] == "dir" and entry[0] != "dir":
                    for child in list(children.pop(path, ())):
                        drop(child, i)  # a file in place of a directory hides what the directory held
                # a link retargeted, a file turned into a directory, a directory's new mode: a change all the same
                shadowed.append((path, old[1], "replaced", i, old[0][0]))
            view[path] = (entry, i)
            # index the path under each of its directories, the ones the layer lists no entry for included
            child, parent = path, posixpath.dirname(path)
            while child and child not in children[parent]:
                children[parent].add(child)
                child, parent = parent, posixpath.dirname(parent)
    return view, shadowed, hiding


def repo_roots(paths, allowed) -> list:
    """Project roots outside the installed and system trees and the allowed source trees: their marker files."""
    return sorted(p for p in paths if posixpath.basename(p) in REPO_MARKERS
                  and not INSTALLED_TREES.search(p + "/") and not under(p, allowed))


def config_findings(config: dict, base: dict, extra=(), generic=()) -> list:
    """(rule, class, where, detail) for the image config, which ships: anyone reads it with docker inspect and docker
    history. Each ENV entry, label, other runtime setting and build step is read for a credential-like ENV name, the
    token formats, a PEM private key and, with ``extra`` (the --names patterns) and ``generic`` (GENERIC_NAMES), a name;
    build arguments reach every RUN step's history line, so a mirror's host given as one is found there. A part the
    base config has too (the same ENV entry or label or setting, the base's own history, which the image's begins
    with) is BASE, never FAIL; in the image's own parts a --names hit FAILs and a generic one is REPORT, as in files."""
    cfg, base_cfg = config.get("config") or {}, base.get("config") or {}
    base_env, base_labels = set(base_cfg.get("Env") or []), base_cfg.get("Labels") or {}
    base_steps = [step.get("created_by") for step in base.get("history") or []]
    parts = [(f"ENV {entry.partition('=')[0]}", entry, entry in base_env) for entry in cfg.get("Env") or []]
    parts += [(f"LABEL {key}", f"{key}={value}", base_labels.get(key) == value)
              for key, value in sorted((cfg.get("Labels") or {}).items())]
    parts += [(f"config {key}", json.dumps(value), base_cfg.get(key) == value) for key, value in sorted(cfg.items())
              if key not in ("Env", "Labels") and value]
    for i, step in enumerate(config.get("history") or []):
        text = step.get("created_by") or ""
        parts.append((f"history {i}", text, i < len(base_steps) and base_steps[i] == text))
    out = []
    for where, text, the_base_s in parts:
        cls, data = "BASE" if the_base_s else "FAIL", text.encode()
        key, _, value = text.partition("=")
        if where.startswith("ENV ") and SECRET_ENV.search(key) and value.strip().lower() not in NOT_A_SECRET | {""}:
            out.append(("token", cls, where, "a credential-like variable set in the image config"))
        out += [("token", cls, where, f"holds {what}") for what, rx in TOKENS.items() if rx.search(data)]
        if PRIVATE_KEY.search(data):
            out.append(("key", cls, where, "holds a PEM private key"))
        for patterns, ours in ((extra, cls), (generic, "BASE" if the_base_s else "REPORT")):
            for what, rx in patterns:
                m = rx.search(data)
                if m:
                    out.append(("names", ours, where, f"{what}: {m.group(0)[:80].decode(errors='replace')}"))
    return out


def config_differences(base: dict, image: dict, except_labels) -> list:
    """(key, the image's value, the base's value) for each runtime setting of the image config that differs from the
    base's (Env, Entrypoint, Cmd, WorkingDir, User, ExposedPorts, Volumes, StopSignal, Healthcheck, Shell, OnBuild),
    and each label but ``except_labels``."""
    a, b = image.get("config") or {}, base.get("config") or {}
    out = [(k, a.get(k), b.get(k)) for k in ("Env", "Entrypoint", "Cmd", "WorkingDir", "User", "ExposedPorts",
                                              "Volumes", "StopSignal", "Healthcheck", "Shell", "OnBuild")
           if a.get(k) != b.get(k)]
    la, lb = a.get("Labels") or {}, b.get("Labels") or {}
    out += [(f"Labels.{k}", la.get(k), lb.get(k)) for k in sorted(set(la) | set(lb))
            if k not in except_labels and la.get(k) != lb.get(k)]
    return out


def owners(view: dict, kept: dict) -> dict:
    """path -> (the package that installs it, the hash the package records for its bytes, as _sums gives it, or ""):
    ``pip:<dist>`` and the ``sha256=`` of each RECORD in the view (pip records none for the bytecode it compiles at
    install, nor for RECORD itself), ``deb:<package>`` from dpkg's file lists and the ``md5=`` of the package's
    md5sums (none for a conffile, which dpkg sums in its status file instead)."""
    md5s = {}  # (dpkg info name, path) -> md5
    for path, data in kept.items():
        if path in view and path.endswith(".md5sums"):
            stem = posixpath.basename(path).removesuffix(".md5sums")
            for row in data.decode("utf-8", "replace").splitlines():
                m = re.match(r"([0-9a-f]{32}) [ *](.+)$", row)  # md5sum's output: the hash, two spaces, the path
                if m:
                    md5s[(stem, norm(m[2]))] = f"md5={m[1]}"
    out = {}
    for path, data in kept.items():
        if path not in view:
            continue
        text = data.decode("utf-8", "replace")
        if path.endswith(".dist-info/RECORD"):
            site, info = posixpath.dirname(posixpath.dirname(path)), posixpath.basename(posixpath.dirname(path))
            dist = info.removesuffix(".dist-info").rsplit("-", 1)[0].lower().replace("_", "-")
            # RECORD is CSV (path, hash, size): a path with a comma or a quote in it is quoted
            for row in csv.reader(io.StringIO(text, newline="")):
                if row and row[0]:
                    out[norm(posixpath.join(site, row[0]))] = (f"pip:{dist}", row[1] if len(row) > 1 else "")
        elif path.endswith(".list"):
            stem = posixpath.basename(path).removesuffix(".list")
            for row in text.splitlines():
                if row.strip():
                    out.setdefault(norm(row), (f"deb:{stem.split(':')[0]}", md5s.get((stem, norm(row)), "")))
    return out


def nixl_findings(view: dict, kept: dict, rules: Rules, add) -> None:
    """NIXL's licence and the notice of its patches (the nixl rule)."""
    infos = sorted({m.group(0) for p in view for m in [re.search(r"^.*?/(?:site|dist)-packages/nixl(?:_cu\d+)?-[^/]+"
                                                                 r"\.dist-info", p)] if m})
    names = {re.sub(r"-[^-]+\.dist-info$", "", posixpath.basename(i)) for i in infos}
    for want in ("nixl", "nixl_cu13"):
        if want not in names:
            add("nixl", want, f"the {want} distribution is not installed", None)
    for info in infos:
        text = kept.get(f"{info}/licenses/LICENSE", b"")
        if not all(m in text for m in APACHE):
            add("nixl", f"{info}/licenses/LICENSE", "missing, or not NIXL's Apache-2.0 licence", None)
    notice = rules.nixl_notice
    shipped = {posixpath.basename(p): d for p, d in kept.items() if posixpath.dirname(p) == notice and p in view}
    notes = b"".join(shipped.get(n, b"") for n in ("README.md", "NOTICE.md"))
    if not notes:
        add("nixl", f"{notice}/NOTICE.md", "no README.md or NOTICE.md about the NIXL build", None)
    for ident in rules.nixl_ids:
        if notes and ident[:8].encode() not in notes:
            add("nixl", f"{notice}/NOTICE.md", f"neither README.md nor NOTICE.md names {ident[:8]}", None)
    for name, data in rules.nixl_patches.items():
        if name not in shipped:
            add("nixl", f"{notice}/{name}", "a patch of patches/nixl the image does not ship", None)
        elif shipped[name] != data:
            add("nixl", f"{notice}/{name}", "differs from the checkout's patches/nixl", None)
    for name in sorted(n for n in shipped if n.endswith(".patch") and n not in rules.nixl_patches):
        add("nixl", f"{notice}/{name}", "a patch the checkout's patches/nixl does not hold", None)


def scan(args) -> int:
    rules = Rules(args)
    base_layer_ids = set(filter(None, args.base_layers.split(",")))
    layers, docs = read_save(sys.stdin.buffer, base_layer_ids, rules)
    base_ids, base_config = image_layers(docs, args.base_id)
    rel_ids, rel_config = image_layers(docs, args.image_id)
    missing = [d for d in base_ids + rel_ids if d not in layers]
    if missing:
        raise ValueError(f"layer {missing[0]} is not in the saved tar (a truncated save?)")
    if base_layer_ids and base_layer_ids != set((base_config.get("rootfs") or {}).get("diff_ids") or []):
        raise ValueError("--base-layers are not the base image's layers (its config's rootfs.diff_ids)")
    shared = set(base_ids)
    rel_layers = [layers[d] for d in rel_ids]
    own = [d not in shared for d in rel_ids]
    # Docker's own store saves each layer uncompressed under its diff ID, so read_save told shared from own layers as it
    # read them. The containerd store saves the compressed blobs, under other digests: every layer was then read as an
    # own one, and the rules only the image's own layers are held to are dropped here from the layers it shares
    for i, layer in enumerate(rel_layers):
        if layer.own and not own[i]:
            layer.findings = [f for f in layer.findings if f[0] not in OWN_RULES]
            layer.names = []
    view, shadowed, hiding = merged_view(rel_layers)
    base_view, _, _ = merged_view([layers[d] for d in base_ids])
    kept = {p: d for layer in rel_layers for p, d in layer.kept.items()}
    findings = []  # (rule, class, layer index or None, path, detail)

    def inherited(path: str, layer_index: int) -> bool:
        entry = rel_layers[layer_index].entries.get(path)
        old = base_view.get(path)
        if entry is None or old is None:
            return False
        if entry[0] == "dir":
            return same_directory(old[0], entry)
        # built FROM the base, a file at a base path is a rewrite, whatever its size
        return args.flattened and old[0][:2] == entry[:2]

    def add(rule, path, detail, layer_index):
        if layer_index is None or rule in ("forbid", "restricted"):
            cls = "FAIL"
        else:
            cls = "FAIL" if own[layer_index] and not inherited(path, layer_index) else "BASE"
        findings.append((rule, cls, layer_index, path, detail))

    plugin_roots = sorted(posixpath.dirname(p) for p in view
                          if re.search(r"(site|dist)-packages/sglang_neuron/__init__\.py$", p))
    who = owners(view, kept)

    def owner_of(path: str) -> tuple:
        """(the package that installs ``path``, the hash it records for the file or ""), (None, "") for a path no
        package installs. A module's bytecode belongs to its source's package, whose hash is the source's."""
        if path in who:
            return who[path]
        m = re.match(r"(.*)/__pycache__/([^/.]+)\.[^/]+\.pyc$", path)
        source = who.get(f"{m[1]}/{m[2]}.py") if m else None
        return (source[0], "") if source else (None, "")

    def package_of(path: str, i: int):
        """The package whose own file layer ``i``'s ``path`` is: its owner, when the package records the hash of these
        very bytes or no hash for the file; None for a path no package installs or a package's file a build step
        wrote over (by its path alone, any step's file at a package's path would get the package's pass)."""
        owner, recorded = owner_of(path)
        return owner if not recorded or recorded in rel_layers[i].sums.get(path, ()) else None

    def over(path: str, i: int) -> str:
        """What a finding adds about a package's file whose bytes in layer ``i`` are not the package's."""
        owner = owner_of(path)[0]
        return f", over {owner}'s file (not the bytes the package records)" if owner and not package_of(path, i) else ""

    for i, layer in enumerate(rel_layers):
        for rule, path, detail in layer.findings:
            if rule in ("key", "token") and detail.startswith("holds "):
                owner = package_of(path, i)
                if owner and owner.startswith("pip:") and owner != "pip:sglang-neuron" and own[i] \
                        and not inherited(path, i):
                    # a published package's own test vectors (pycryptodome's keys, transformers' dummy token)
                    findings.append((rule, "REPORT", i, path, f"{detail}, in {owner}'s own files"))
                    continue
                detail += over(path, i)
            add(rule, path, detail, i)
        for path in layer.whiteouts + layer.opaque:
            if (i, path) not in hiding:
                continue  # it hides nothing below it: the marker of a directory this layer creates
            if rules.forbid and under(path, rules.forbid):
                add("forbid", path, "a whiteout of a --forbid path (its bytes are in a layer below)", i)
            if own[i] and rules.own_only and not under(path, rules.own_only):
                add("own-only", path, "a whiteout outside every --own-only path", i)
    all_paths = {p: i for i, layer in enumerate(rel_layers) for p in layer.entries}
    for marker in repo_roots(all_paths, rules.allow_source + rules.forbid):
        root = posixpath.dirname(marker)
        subdirs = [d for d in REPO_SUBDIRS if any(under(p, [posixpath.join(root, d)]) for p in all_paths)]
        add("source", marker, "a project root" + (f" with {', '.join(subdirs)}" if subdirs else ""),
            all_paths[marker])
    for path, index, how, by, kind in shadowed:
        if how == "deleted" and kind == "file":  # a link or a directory holds no bytes to recover
            add("whiteout", path, f"deleted by layer {by}, its bytes still in this layer", index)
        if args.adds_only and own[by] and not own[index]:
            add("replace", path, f"{how} by own layer {by} (the {kind} of layer {index})", by)
    findings += [(rule, cls, None, where, detail)
                 for rule, cls, where, detail in config_findings(rel_config, base_config, rules.extra, rules.names)]
    if args.layer_prefix and rel_ids[:len(base_ids)] != base_ids:
        add("prefix", "layers", "the base's layers are not the first layers of the image, in order", None)
    if args.same_config:
        for key, ours, theirs in config_differences(base_config, rel_config, args.same_config_except):
            add("config", f"config {key}", f"differs from the base's: {theirs!r} -> {ours!r}"[:300], None)
    for path in args.require:  # a file, or a directory with something in it (a layer need not list the directory)
        if norm(path) not in view and not any(p.startswith(norm(path) + "/") for p in view):
            add("require", norm(path), "missing from the final file system", None)
    elf_owners = collections.Counter()
    for i, layer in enumerate(rel_layers):
        for path in layer.elf:
            if not own[i] or inherited(path, i) or view.get(path, (None, -1))[1] != i:
                continue
            owner = owner_of(path)[0]
            elf_owners[owner or "unowned"] += 1
            if owner is None or owner == "pip:sglang-neuron" or under(path, plugin_roots + rules.allow_source):
                add("elf", path, f"a shared object or program {('of ' + owner) if owner else 'no package owns'}", i)
            elif owner == "pip:sglang":
                findings.append(("elf", "REPORT", i, path, "a binary in SGLang's own wheel"))
        for path, what, text, extra in layer.names:
            if inherited(path, i):  # every version of the file: one a later layer replaces ships in this layer
                continue
            owner = package_of(path, i)
            if extra and not (owner and owner != "pip:sglang-neuron"):
                # our content: the plugin's distribution, files no package owns, the checkout, what a build step
                # wrote over a package's file
                cls, where = "FAIL", over(path, i)
            else:
                # a published package's own file (an author's name, an upstream contributor's login): reviewed, ours
                # to report, not to fail; a generic pattern (doc examples hold private addresses) is always REPORT
                cls, where = "REPORT", f", in {owner}'s own files" if owner else over(path, i)
            findings.append(("names", cls, i, path, f"{what}: {text}{where}"))
    if rules.nixl_patches:
        nixl_findings(view, kept, rules, add)

    write_outputs(args.out, view, base_view, rel_ids, rel_layers, own, findings, plugin_roots, elf_owners, shadowed)
    fails = [f for f in findings if f[1] == "FAIL"]
    added = sum(1 for p in view if p not in base_view)
    print(f"layers: {len(rel_ids)}, {sum(own)} the image's own; files: image {len(view)}, base {len(base_view)}, "
          f"added {added}")
    print(f"installed sglang_neuron: {', '.join('/' + p for p in plugin_roots) or 'none'}; ELF files the image adds, "
          f"by owner: {dict(elf_owners.most_common(8))}")
    counts = collections.Counter((f[0], f[1]) for f in findings)
    print("findings, rule FAIL/REPORT/BASE: " + (", ".join(
        f"{r} {counts[(r, 'FAIL')]}/{counts[(r, 'REPORT')]}/{counts[(r, 'BASE')]}"
        for r in sorted({f[0] for f in findings})) or "none"))
    for f in fails[:args.show]:
        print(f"  {_line(f, rel_ids)}")
    if len(fails) > args.show:
        print(f"  ... {len(fails) - args.show} more in {args.out}/findings.txt")
    return 1 if fails else 0


def _line(f, rel_ids) -> str:
    rule, cls, index, path, detail = f
    where = f"layer {index:2d} {rel_ids[index][7:19]}" if index is not None else "config or view       "
    shown = path if path.startswith(("ENV ", "LABEL ", "config ", "history")) else "/" + path
    return f"{cls:6s} {rule:8s} {where}  {shown}: {detail}"


def write_outputs(out, view, base_view, rel_ids, rel_layers, own, findings, plugin_roots, elf_owners, shadowed):
    os.makedirs(out, exist_ok=True)
    added = sorted(p for p in view if p not in base_view)
    changed = sorted(p for p in view if p in base_view and view[p][0][:2] != base_view[p][0][:2])
    _write_list(os.path.join(out, "release_files.txt"), view, sorted(view))
    _write_list(os.path.join(out, "base_files.txt"), base_view, sorted(base_view))
    _write_list(os.path.join(out, "added_files.txt"), view, added)
    _write_list(os.path.join(out, "changed_files.txt"), view, changed)
    outside = [p for p in added if not re.match(r"^(root|home)(/|$)", p)]
    _write_list(os.path.join(out, "added_outside_home.txt"), view, outside)
    with open(os.path.join(out, "own_layer_paths.txt"), "w") as f:
        for i, layer in enumerate(rel_layers):
            if own[i]:
                f.write("".join(f"{i}\t/{p}\n" for p in sorted(layer.entries)))
                f.write("".join(f"{i}\t/{p}\twhiteout\n" for p in sorted(layer.whiteouts + layer.opaque)))
    with open(os.path.join(out, "sglang_neuron_path.txt"), "w") as f:
        f.write("".join(f"/{p}\n" for p in plugin_roots))
    order = {"FAIL": 0, "REPORT": 1, "BASE": 2}
    with open(os.path.join(out, "findings.txt"), "w") as f:
        for row in sorted(findings, key=lambda r: (order[r[1]], r[0], r[3])):
            f.write(_line(row, rel_ids) + "\n")
    groups, sizes = collections.Counter(), collections.Counter()
    for p in added:
        top = "/" + "/".join(p.split("/")[:2])
        groups[top] += 1
        sizes[top] += view[p][0][1]
    layer_rows = [{"index": i, "digest": d, "from": "image" if own[i] else "base", "entries": len(layer.entries),
                   "bytes": layer.bytes, "whiteouts": len(layer.whiteouts) + len(layer.opaque),
                   "xattr_files": layer.xattr_files} for i, (d, layer) in enumerate(zip(rel_ids, rel_layers))]
    with open(os.path.join(out, "layers.json"), "w") as f:
        json.dump(layer_rows, f, indent=1)
    report = {
        "layers": layer_rows,
        "files": {"image": len(view), "base": len(base_view), "added": len(added), "changed": len(changed),
                  "added_outside_home": len(outside),
                  "replaced_in_own_layers": sum(1 for s in shadowed if s[2] == "replaced" and s[4] == "file"
                                                and own[s[1]])},
        "added_by_directory": {k: {"files": groups[k], "bytes": sizes[k]} for k in sorted(groups)},
        "plugin": ["/" + p for p in plugin_roots],
        "elf_by_owner": dict(elf_owners),
        "findings": [{"class": c, "rule": r, "layer": i, "path": p, "detail": d}
                     for r, c, i, p, d in findings if c != "BASE"],
        "base_findings": dict(collections.Counter(r for r, c, *_ in findings if c == "BASE")),
        "hashes": {"/" + p: rel_layers[view[p][1]].hashes[p] for p in sorted(view)
                   if p in rel_layers[view[p][1]].hashes},
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("added over the base, by directory (files, MB):")
    for top in sorted(groups, key=lambda k: -sizes[k])[:25]:
        print(f"  {top:44s} {groups[top]:8d} {sizes[top] / 1e6:10.1f}")


def _write_list(path: str, view: dict, paths) -> None:
    with open(path, "w") as f:
        for p in paths:
            (kind, size, mode, owner, link, xattrs), index = view[p]
            f.write(f"/{p}\t{kind}\t{size}\t{mode:o}\t{owner}\t{link}\t{xattrs}\tlayer {index}\n")


def versions(_args) -> int:
    """Run inside the image: what it installs, as JSON on stdout."""
    import importlib.metadata as md
    import importlib.util
    import platform

    wanted = re.compile(r"^(torch|torch-neuronx|torchvision|neuronx-cc|nki|libneuronxla|neuron.*|aws-neuron.*|"
                        r"sglang|sglang-neuron|nixl.*|transformers|safetensors|triton)$")
    dists = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if wanted.match(name):
            row = {"version": dist.version, "location": str(dist.locate_file(""))}
            direct = dist.read_text("direct_url.json")
            if direct:
                info = json.loads(direct)
                vcs = info.get("vcs_info") or {}
                row["source"] = {k: v for k, v in (("url", info.get("url")), ("commit", vcs.get("commit_id")),
                                                   ("editable", (info.get("dir_info") or {}).get("editable")))
                                 if v}
            dists[name] = row
    debs = {}
    try:
        out = subprocess.run(["dpkg-query", "-W", "-f", "${Package}\t${Version}\n", "aws-neuronx-*"],
                             capture_output=True, text=True, check=False).stdout
        debs = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line and line.split("\t")[1])
    except OSError:
        pass
    os_release = {}
    if os.path.exists("/etc/os-release"):
        for line in open("/etc/os-release"):
            key, _, value = line.strip().partition("=")
            if key in ("PRETTY_NAME", "VERSION_ID"):
                os_release[key.lower()] = value.strip('"')
    sglang = None
    spec = importlib.util.find_spec("sglang")
    if spec is not None and spec.origin:
        root = os.path.dirname(spec.origin)
        # what sglang.__version__ is, read without importing sglang: its version.py takes _version.py (setuptools-scm
        # writes it at build time), then the installed distribution's version; its last fallback, 0.0.0.dev0, is no
        # version at all, so a tree with neither says None
        try:
            version = re.search(r"^__version__\s*=\s*(?:version\s*=\s*)?['\"]([^'\"]+)",
                                open(os.path.join(root, "_version.py")).read(), re.M)
            version = version.group(1) if version else None
        except OSError:
            version = (dists.get("sglang") or {}).get("version")
        sglang = {"path": root, "version": version}
    print(json.dumps({"python": platform.python_version(), "os": os_release, "pip": dict(sorted(dists.items())),
                      "apt": dict(sorted(debs.items())), "sglang_tree": sglang}, indent=1))
    return 0


def content_limit(value: str) -> int:
    """--content-limit: at least TEXT_SNIFF, so a file's first window holds all the bytes the text sniff reads."""
    limit = int(value)
    if limit < TEXT_SNIFF:
        raise argparse.ArgumentTypeError(f"{limit} is below the text sniff's {TEXT_SNIFF} bytes")
    return limit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="audit_release_image.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="scan `docker save <base> <image>` from stdin")
    s.add_argument("--base-id", required=True, help="the base image's id (docker image inspect -f {{.Id}})")
    s.add_argument("--image-id", required=True, help="the audited image's id")
    s.add_argument("--base-layers", default="", help="the base's layer digests, comma-separated (RootFS.Layers)")
    s.add_argument("--out", required=True, help="a directory for the lists and findings")
    s.add_argument("--forbid", action="append", default=[], help="a path no layer may hold (repeatable)")
    s.add_argument("--restrict", action="append", default=[],
                   help="a directory where a layer may hold only the --restrict-allow paths (repeatable)")
    s.add_argument("--restrict-allow", action="append", default=[],
                   help="a path a --restrict directory may hold, with everything under it (repeatable)")
    s.add_argument("--require", action="append", default=[], help="a path the image must hold (repeatable)")
    s.add_argument("--allow-source", action="append", default=[], help="a source tree the image ships on purpose")
    s.add_argument("--leftover", action="append", default=[], help="a build-leftover path regex (own layers)")
    s.add_argument("--own-only", action="append", default=[], help="the only path own layers may add (repeatable)")
    s.add_argument("--adds-only", action="store_true", help="own layers may not change or delete a lower entry")
    s.add_argument("--nixl-patches", help="the checkout's patches/nixl: the image must ship NIXL with these patches")
    s.add_argument("--nixl-notice", default="/usr/share/doc/sglang-neuron/nixl", help="where the image ships them")
    s.add_argument("--nixl-ids", help="commits the notice must name, comma-separated (build_nixl.sh's)")
    s.add_argument("--names", help="a file of further internal-name patterns, one regular expression a line")
    s.add_argument("--content-limit", type=content_limit, default=CONTENT_LIMIT,
                   help="read files up to this many bytes whole, and longer ones in windows of this size")
    s.add_argument("--base-content", action="store_true", help="search the bytes of the shared base layers too")
    s.add_argument("--hash-under", action="append", default=[], help="report the sha256 of the files under a path")
    s.add_argument("--layer-prefix", action="store_true", help="the base's layers must open the image's, in order")
    s.add_argument("--flattened", action="store_true",
                   help="the image holds the base's files in its own layers: an own file as the base has it is BASE")
    s.add_argument("--same-config", action="store_true", help="the image config must equal the base's")
    s.add_argument("--same-config-except", action="append", default=[], help="a label --same-config ignores")
    s.add_argument("--show", type=int, default=40, help="failures printed (all are in findings.txt)")
    s.set_defaults(run=scan)
    v = sub.add_parser("versions", help="(inside the image) the installed versions, as JSON")
    v.set_defaults(run=versions)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, KeyError, OSError, tarfile.TarError, RuntimeError, re.error, csv.Error) as error:
        print(f"audit_release_image: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

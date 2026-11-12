# NIXL patches

Disaggregated serving moves the KV cache from prefill to decode servers with NIXL's LIBFABRIC backend over EFA. The
servers load NIXL 1.5.0 built from source with the patches here, all to its libfabric plugin. 0001 to 0005 are NIXL
commits applied unchanged; 0006 to 0009 are this project's own fixes, not proposed upstream yet.

## Why a source build

NIXL's PyPI wheels cannot be used as they are, for two reasons:

- Their libnixl links etcd's C++ client, which brings gRPC and a protobuf that collides with the protobuf torch-neuronx
  links statically. A process that imports both crashes at startup. SGLang's KV transfer does not use NIXL's etcd
  metadata exchange, so the build leaves etcd out.
- Their libfabric plugin hangs transfers under a disaggregated load and grows a table with every request (0004, 0007
  and 0008 below).

## Patches

| Patch | Upstream | Why this project carries it |
|---|---|---|
| 0001 AMD accelerator detection | [#1750](https://github.com/ai-dynamo/nixl/pull/1750) | Context only: it does nothing on a Neuron host, and 0005 does not apply without it. |
| 0002 notify the target of an unposted batch write | [#2107](https://github.com/ai-dynamo/nixl/pull/2107) | Context only: it acts on notified transfers, which SGLang does not post, and 0005 does not apply without it. |
| 0003 name `fi_writemsg` in messages | [#2159](https://github.com/ai-dynamo/nixl/pull/2159) | Context only: log messages, and 0005 does not apply without it. |
| 0004 serialize endpoint calls | [#2308](https://github.com/ai-dynamo/nixl/pull/2308) | The progress thread and the caller's thread use a rail's endpoint at once; on EFA that corrupts the provider's state and hangs transfers. |
| 0005 identify senders by source address | [#2294](https://github.com/ai-dynamo/nixl/pull/2294) | Lifts the limit of 255 peers per agent, which a prefill whose decode servers have more ranks than that between them reaches. 0007 and 0008 are written against the code it brings. |
| 0006 keep EFA unsolicited write receive on | not proposed | Without it a decode rank that takes many KV writes at once runs out of posted receives, its endpoint fails, and the decode server stops. |
| 0007 fail the request of every CQ error entry | not proposed | One of the plugin's two completion readers drops error entries, so the failed transfer never completes and the prefill's transfer worker waits on it forever, with every later transfer behind it. |
| 0008 unnotified writes are not counted | not proposed | The target kept a pending entry, counting its writes, for every transfer without a notification, which is every SGLang KV transfer, so the table grew with every request and slowed each completion the decode's progress threads handle. Upstream main's [#2436](https://github.com/ai-dynamo/nixl/pull/2436), after the base, removed the slowdown (it checks only the entry an event updated) but not the entries, which it lists as a follow-up. |
| 0009 drop a failed transfer's notification | not proposed | The target delivered the notification of a transfer that failed on the initiator once its other writes had landed, so a partly written transfer looked complete. With 0009 the PP seam takes a hop's notification as its proof; without it each hop also waits for its sender's commit (see the NIXL notes in `sglang_neuron/distributed/pp_seam.py`). |

Each patch file starts with what it changes, why, its upstream status and its tests. The five backports are in
upstream `main` (0001 to 0003 also in the v1.5.0 tag, which is not on the release/1.5.0 branch the base and the PyPI
1.5.0 wheels come from; 0004 in v1.5.2) and go away when the base moves to a release that has them.

Built against the EFA installer's libfabric headers, as `tools/build_nixl.sh` builds it, upstream's plugin turns EFA
unsolicited write receive off on every endpoint (0006 keeps it on). The PyPI 1.5.0 wheels were built against headers
that lack the option, so the feature stays on there, which is why 0006 is not among their defects above; from v1.5.2
upstream turns it off whatever the headers. Upstream turns it off because, with it on, remote write completions can be
lost under load, and with them the notifications that ride on them. SGLang's KV transfers do not depend on those: they
carry no notification, and the prefill learns each transfer's outcome from its own transfer handles and reports it to
the decode over ZMQ.

## Building

In the serving container (it has git, gcc, g++, patchelf, binutils and the EFA headers under `/opt/amazon/efa`), with
the Python the servers run first on `PATH`: the script builds for the `python3` it finds there, and names the wheel
after the CUDA major of that Python's torch (`nixl_cu13` for the container's torch 2.14.0+cu130), the module NIXL's
meta package imports. `<out dir>` must not exist.

```bash
NIXL_WHEELS=<dir> bash tools/build_nixl.sh <out dir>
```

`NIXL_WHEELS` holds the build tools and the `nixl` 1.5.0 meta wheel that `tools/build_nixl.sh` pins by sha256
(`TOOL_WHEELS` and `META_WHEEL`). On a machine with PyPI, with those lines in `req.txt`:

```bash
pip download --no-deps --require-hashes -r req.txt --only-binary=:all: --implementation cp --python-version 3.14 \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform any \
  -d <dir>
```

`NIXL_PACKAGECACHE` holds NIXL's meson subproject archives for a node that cannot reach their hosts: the source and
patch archives that the cloned NIXL's `subprojects/*.wrap` name, with the hashes they give. The abseil-cpp and asio
patch archives come only from wrapdb.mesonbuild.com; the sources also have GitHub mirrors. NIXL names taskflow by git
tag (a `[wrap-git]` with no hash), which a cache cannot serve, so the script replaces that wrap with the same release's
GitHub archive at its sha256 (`TASKFLOW_WRAP`; it refuses a clone whose wrap names another tag). `meson subprojects
download abseil-cpp asio liburing taskflow tomlplusplus`, run in a clone that has the replaced wrap, leaves exactly
these archives in `subprojects/packagecache`. With them and `NIXL_REPO` naming a local clone, the build needs no
network.

The script clones NIXL at the `release/1.5.0` commit the PyPI wheels were built from, applies the patches, refuses a
source tree other than the pinned one, builds one wheel with only the LIBFABRIC plugin and without etcd or the
Prometheus exporter, and installs it with the meta wheel into a staging overlay. It checks that nothing in the overlay
needs or exports etcd, gRPC or protobuf, that it has no Prometheus exporter and that the plugin carries 0006, 0007 and
0009, then moves it to `<out dir>` and prints the plugin's md5; a failed build is kept in `<out dir>.build.<suffix>`.
The same container gives the same plugin on every node.

## Configuration

- Put `<out dir>` first on `PYTHONPATH` of both servers, and leave `NIXL_PLUGIN_DIR` unset so that libnixl loads the
  plugin beside it.
- Both ends of a transfer must run this build: 0005 and 0008 version the connection handshake, and a peer of another
  version is refused at connection setup.
- The platform sets `NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV=1` (0006) on both disaggregated roles and refuses a
  launch value that does not read as true (`true`, `yes` and `y` are passed on as `1`), since the EFA provider silently
  falls back to slower emulated writes between two ends that disagree.

## Validation

0001, 0005 and 0007 add unit tests to NIXL's own suite. 0007's `cq_error_test` needs a debug build of the patched
source (`meson setup -Dbuildtype=debug -Dbuild_tests=true`, then `meson test cq_error_test`).

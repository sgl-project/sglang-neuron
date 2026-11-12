# SGLang-Neuron container image

One Dockerfile builds two images from the same commit:

```
  Neuron SDK 2.33.0 container (by digest)
  + SGLang at the pinned commit, its runtime dependencies, torchvision (CPU)
  + SGLang Diffusion's packages (docker/diffusion-requirements.txt)
  + sglang_neuron
  + NIXL 1.5.0 with patches/nixl
  + the repository's public part in /opt/sglang-neuron           -> public  (the image that serves)
  + the repository's checkout in /opt/sglang-neuron/src          -> qa      (public's layers, unchanged, plus these)
```

`public` holds no other file of the repository checkout in any layer, so it needs no stripping or flattening: under
/opt/sglang-neuron the gate allows only the repository's public part (README.md, LICENSE, pyproject.toml, docs/,
examples/, sglang_neuron/, the NIXL patches with their README, and tools/glm52/build_latent_store.py, the GLM-5.2 1M
example's weight-store builder; never tests/, docker/ or scripts/). The build check proves it the commit's files, and
the build check and the gate prove its sglang_neuron/ the installed package. `qa` is built from `public` and only adds,
so QA runs against exactly the layers that serve.

Neuron publishes the serving image in the Amazon ECR Public Gallery as
[pytorch-inference-sglang-neuron](https://gallery.ecr.aws/neuron/pytorch-inference-sglang-neuron):

```
docker pull public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
```

To run it, see the [container guide](../docs/guides/container.md). The rest of this page builds the images yourself from
a checkout, on the host that runs them (or another Linux x86_64 host). The build host needs:

- Linux on x86_64, with GNU coreutils (`build.sh` stops at once without them) and bash 4.2 or later;
- Docker Engine 23.0 or later (its `docker build` is buildx), with either image store: the classic one (overlay2) or
  containerd; the daemon may be local or reached through `DOCKER_HOST`;
- Python 3.11 or later (`build.sh` reads the release pin with `tomllib`) and git;
- about 60 GB free under Docker's root (`build.sh` warns below that) and network access during the build
  ([Build](#build)).

```
git clone <this repository> && cd <checkout> && git checkout <commit>
docker/build.sh            # build both and gate them
```

The images are `sglang-neuron:<tag>` (public) and `sglang-neuron:<tag>-qa`; `build.sh` prints both when it ends, and
writes its logs, the gate's reports (`audit/`) and `versions.json` to `build/image-<tag>-<time>/` in the checkout
(ignored by git). The [container runbook](../docs/guides/container-runbook.md) goes from there to a
serving model (one node, and GPT-OSS-120B or GLM-5.2 in prefill/decode) and a QA run.

## Build

```
docker/build.sh            # build both and gate them
```

`build.sh` exports `HEAD` with `git archive`, so local edits never reach an image, and names the images
`sglang-neuron:<tag>` and `sglang-neuron:<tag>-qa` in the local Docker image store; it pushes nothing. The
Dockerfile's `public` stage runs `docker/check_image.py`, which fails the build unless SGLang and the plugin are
installed packages at the pinned commits, the plugin's torch and torch-neuronx pins hold, and the NIXL the image
imports is the patched build. `docker/verify_images.sh` then reads every layer of both images
(`tools/audit_release_image.sh`): `qa` is `public` plus additions only, no layer of `public` holds the checkout, none of
its own layers holds a build leftover, the image's own content (the plugin, the NIXL notice, any file no package owns)
holds no key or token and, with `--names <file>`, no listed name (one regular expression a line). A key or token in a
published Python package's files (test keys), or a listed name in any package's files (authors' names), is reported
for review rather than failed. The build directory gets the logs, the gate's reports and, written last, `versions.json`:
the commit, the gated images' ids and the versions inside. A build that fails writes no `versions.json`.

| option | |
|---|---|
| `--tag <tag>` | the public image's tag (default `v<version>-<commit>-dev`, or `v<version>-<commit>` with `--release`); the QA image's is `<tag>-qa` |
| `--release` | a release build: needs `--names` and a clean working tree |
| `--names <file>` | the internal-name patterns; optional, required with `--release` (without it the gate says it did not check) |
| `--torch-neuronx <wheel>` | install that torch-neuronx wheel in place of the container's ([below](#moving-to-another-torch-neuronx-build)) |
| `--no-cache` | rebuild every stage instead of reusing Docker's build cache |
| `--out <dir>` | the build directory, which must not exist (default `build/image-<tag>-<time>/` in the checkout) |

To run the images on other hosts, build them there, copy them (`docker save sglang-neuron:<tag> | ssh <host>
docker load`), or push them to a registry of your own with `docker tag` and `docker push`: `versions.json` names the
ids the gate passed. For a release build (`--release --names <file>`), keep the build directory with the release
record: `versions.json` names the commit and the patterns file's sha256, and `audit/` holds the gate's reports.

The build needs network access: public.ecr.aws for the Neuron SDK base image (about 27 GB, pulled anonymously by
digest; if that pull fails with an authorization error, a stored public.ecr.aws login has expired: `docker logout
public.ecr.aws`), PyPI, GitHub, PyTorch's CPU wheel index and NIXL's build dependencies (wrapdb.mesonbuild.com,
SourceForge). The images themselves need none. `patches/nixl/README.md` describes
building NIXL alone with no network (`NIXL_REPO`, `NIXL_PACKAGECACHE`).

Every Python dependency the images install, and every tool their wheels are built with, is pinned by version and
sha256 (`docker/requirements-lock.txt`, `docker/torchvision-lock.txt`, `docker/qa-requirements-lock.txt`,
`docker/wheel-build-lock.txt`), so a build resolves nothing. SGLang's own
dependency list is its GPU stack (a CUDA torch, FlashInfer, its CUDA kernels), so the image installs SGLang without
dependencies and locks the list SGLang gives for platforms with their own torch
(`tools/sglang_runtime_requirements.py`) with SGLang Diffusion's packages (`docker/diffusion-requirements.txt`), so the
public image serves SGLang Diffusion's models too; the QA lock adds the QA image's requirements
(`docker/qa-requirements.txt`). The locks are generated, never edited:

```
docker/lock_requirements.sh                           # after moving SGLang's commit or a requirement: the rest stays
docker/lock_requirements.sh --upgrade-package <name>  # move one package (repeatable) to its newest release
docker/lock_requirements.sh --upgrade                 # move every package
```

It runs in the base image and keeps every locked version unless told otherwise: a package moved with
`--upgrade-package` moves as far as the other locked versions allow, so name each dependency that must move with it.
It refuses to replace a package of the base image other than the ones it names with a reason (`BASE_CHANGES` in
`docker/lock_requirements.py`: tokenizers and huggingface-hub, which SGLang's pin needs). Review the diff, then rebuild
and re-test.

## Run

```
docker run --init --ipc=host --cap-add SYS_NICE \
    $(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done) \
    -v <models>:/models -p 127.0.0.1:30000:30000 \
    sglang-neuron:<tag> python3 -m sglang.launch_server --model-path /models/<model> --host 0.0.0.0 --port 30000 ...
```

- Pass every Neuron device the server uses (the loop above passes all of the host's), as the Neuron container
  documentation describes.
- `--host 0.0.0.0` makes the server reachable from outside its container, and `-p 127.0.0.1:30000:30000` publishes
  it on the host's loopback only (`curl http://127.0.0.1:30000/health`). Publish it on another address only for
  clients that need it.
- `--ipc=host` (or a large `--shm-size`): SGLang's processes exchange tensors through `/dev/shm`, and Docker's
  default is 64 MB.
- A disaggregated (prefill/decode) or multi-node deployment also needs the EFA devices (`--device
  /dev/infiniband/uverbs<N>` for each, with `--cap-add IPC_LOCK --ulimit memlock=-1`) and a network the peers reach each
  other on, usually `--network host`. With `--network host` the container uses the host's network, so `-p` does not
  apply and `--host 0.0.0.0` listens on every interface of the host: give `--host` the address the peers and the
  balancer reach the server on.
- `--init` reaps the server's worker processes, and `docker stop` sends SIGTERM to the server itself (the image has
  no wrapper script in between). Give it time to stop every rank: `docker stop -t 120` (a multi-rank server can take
  longer than the 10 s after which Docker kills it by default), or `--stop-timeout 120` on `docker run`.
- The image sets no Docker health check (the base image's checks a port SGLang does not serve). Probe
  `GET /health` on the server's port instead.
- Prefill and decode servers of a disaggregated deployment must both run this image: the KV transfer uses the patched
  NIXL, whose handshake an unpatched NIXL does not accept. Clients reach the pair through the image's load balancer,
  which sends each request to both halves:
  `python3 -m sglang_neuron.disaggregation.balancer --prefill http://<prefill>:30000 --decode http://<decode>:30000`
  (each model's recipe in `docs/model-recipes/` gives its servers' flags). It needs no Neuron device, so it can run
  in its own container of the image, on any host that reaches both servers:
  `docker run --rm --network host sglang-neuron:<tag> python3 -m sglang_neuron.disaggregation.balancer --prefill ...
  --decode ...`.
  It listens on 127.0.0.1:8000; give it `--host <address>` (and `--port`) for clients on other hosts, and expose it
  only where they need it.
- Mount models and data anywhere but `/workspace` (see Known limitations).
- Keep the compile caches on a persistent mount, or every new container compiles from scratch:
  `-v <cache dir>:/cache -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e
  TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki` (the programs, their graphs and the NKI
  kernels). Mount it at the same path every time (the cache path is part of what is compiled, so a cache moved
  elsewhere recompiles), and use one directory per image version. A launch refuses a program or graph cache built
  from other NKI kernel sources or another kernel library version than its own; a cache from another version with the
  same kernels is accepted (torch-neuronx's own cache key covers the rest of what a program is built from).
- The host needs the Neuron driver Neuron SDK 2.33.0 lists (`aws-neuronx-dkms` 2.31.7.0 or later) and, for multi-node
  serving, EFA.
- `docker inspect -f '{{json .Config.Labels}}' sglang-neuron:<tag>` shows what an image was built from: the plugin
  revision, the Neuron SDK and the SGLang commit.

## Known limitations

- **On a host without Neuron devices** (to read `--help`, say), set `TORCH_DEVICE_BACKEND_AUTOLOAD=0
  NEURON_PLATFORM_TARGET_OVERRIDE=trn3 NEURON_LOGICAL_NC_CONFIG=2`: the container's torch-neuronx loads with torch and
  needs a device otherwise.
- **Do not mount a volume at `/workspace`.** The Neuron SDK 2.33.0 container this image is built on installs
  torch-neuronx (and torchtitan-neuron) in editable mode from `/workspace/torch_neuronx` (and
  `/workspace/torchtitan_neuron`). A volume mounted at `/workspace`, such as `docker run -v $PWD:/workspace ...`, hides
  them, and `import torch_neuronx` then fails with `ModuleNotFoundError`. Mount models, data and work directories
  elsewhere, e.g. `/models` or `/data`. Do not start Python from `/workspace` either: there, `import torch_neuronx`
  finds the checkout directory first and gives an empty namespace package. For torch-neuronx both go away when the
  image installs a torch-neuronx wheel (below); torchtitan-neuron stays editable there, and SGLang does not use it.

- **Give the container `CAP_SYS_NICE`** (`--cap-add SYS_NICE`) under Docker's default seccomp profile. The base's
  libnuma (2.0.19) asks the kernel for the NUMA policy in a way that profile blocks without it, and then reports NUMA as
  unavailable, so NIXL cannot map its network rails to NUMA nodes.
- **SGLang's `/v1/responses` endpoint is off.** SGLang's runtime set for platforms with their own torch, which the
  image installs, pins `openai==2.6.1` (its GPU requirements ask for 2.8.0), whose types predate the Responses API
  code it ships, so the server logs that it disabled the endpoint at startup. `/v1/chat/completions` and
  `/v1/completions` are unaffected.
- **Do not `pip install`, upgrade or reinstall `nixl` or `nixl-cu13`.** The image's NIXL is the patched build
  (`/usr/share/doc/sglang-neuron/nixl/`); a PyPI wheel replaces it and the disaggregated servers then fail. For the
  same reason, `pip check` reports `nixl 1.5.0 requires nixl-cu12`: NIXL's meta package names both of its CUDA builds,
  and the image holds the one that matches its torch.
- **`pip check` lists about 30 `sglang ...` lines**, such as `requires flashinfer-python` and `has requirement
  torch==2.14.1`: SGLang's own requirements are its GPU stack, so the image installs SGLang without them and installs
  its runtime set for other platforms instead (`docker/check_image.py` checks that set at every build).

## Moving to another torch-neuronx build

The image uses the container's own torch-neuronx (2.14.3.0.1, built from the public v2.33.0 source). To build on a
torch-neuronx wheel instead:

```
docker/build.sh --torch-neuronx <path>/torch_neuronx-<version>-cp314-*.whl
```

The `public` stage then uninstalls the container's editable torch-neuronx and installs the wheel (without
dependencies) in site-packages, and `versions.json` records the wheel's name and sha256. `docker/check_image.py` fails
the build if the wheel's requirements are not met or the plugin's torch-neuronx source pins
(`sglang_neuron/backend/torch_pins.py`, `executor_cache.py`, `direct_submit.py`) do not hold on it; update the pins
only after re-validating the new build on Neuron hardware.

## QA

`sglang-neuron:<tag>-qa` is the public image plus the commit's checkout at `/opt/sglang-neuron/src` and the QA
dependencies (`docker/qa-requirements.txt`: pytest and its plugins, lm-eval with its OpenAI-compatible client,
torchmetrics and boto3). The build proves the checkout's `sglang_neuron/` byte-identical to the installed package. A
server started in the QA image imports the installed package, so do not put the checkout on `PYTHONPATH`: the server
would then run the checkout and compile at its path instead of the shipped one.

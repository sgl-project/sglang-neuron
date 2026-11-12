"""Import the SGLang modules a Neuron server process loads; report the first failure per module, and exit 1 if any
fails.

--diffusion adds SGLang Diffusion's server and generator and the plugin's diffusion pipelines. They need the diffusion
packages (docker/diffusion-requirements.txt), which the image installs and an LLM environment
(scripts/setup_sglang_main_env.sh) does not.
"""
import argparse
import importlib
import sys
import traceback

LLM = [
    "sglang.launch_server",
    "sglang.srt.entrypoints.http_server",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.detokenizer_manager",
    "sglang.srt.speculative.eagle_worker_v2",
    "sglang.srt.constrained.xgrammar_backend",
]
DIFFUSION = [
    "sglang.multimodal_gen.runtime.entrypoints.http_server",
    "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator",
    "sglang_neuron.diffusion.plugin",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--diffusion", action="store_true",
                        help="also SGLang Diffusion's entry points and the plugin's diffusion pipelines")
    args = parser.parse_args(argv)
    failed = 0
    for name in LLM + (DIFFUSION if args.diffusion else []):
        try:
            importlib.import_module(name)
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            last = traceback.extract_tb(exc.__traceback__)[-1]
            print("FAIL", name, f"{type(exc).__name__}: {exc}"[:300], f"@ {last.filename}:{last.lineno}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

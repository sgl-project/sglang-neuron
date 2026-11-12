# SPDX-License-Identifier: Apache-2.0
import logging
import os
import threading
import torch
import concurrent.futures
import queue
import time
from typing import NamedTuple, TYPE_CHECKING

from safetensors import safe_open

from sglang_neuron.utils.weight_loader import (
    get_weight_loader,
    SafetensorsWeightLoader,
)

if TYPE_CHECKING:
    from safetensors import PySafeSlice

logger = logging.getLogger(__name__)

# Retry configuration for the page-cache read loop. A read from a network file
# system (for example FSx or Lustre) can fail with a transient OSError, for
# example "Broken pipe".
_PAGE_CACHE_READ_MAX_ATTEMPTS = 4
_PAGE_CACHE_READ_BASE_BACKOFF_S = 1.0
_PAGE_CACHE_READ_MAX_BACKOFF_S = 8.0


class CheckpointLoadResult(NamedTuple):
    """A sharded load's result, as torch's load_state_dict reports it: the loaded tensors by parameter name, and the
    checkpoint keys no mapping references."""

    state_dict: dict[str, torch.Tensor]
    unexpected_keys: list[str]


# How often the loader's main thread looks for files another rank has cached (cached_files_store), and for failed
# worker futures: each look is one store request per file not yet cached, on every rank.
_CACHED_FILE_POLL_S = 0.05


def _safetensors_files(checkpoint_dir: str) -> list[str]:
    """The sorted .safetensors file names of a local checkpoint directory; anything else is refused."""
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir!r} is not a local checkpoint directory: SGLang-Neuron loads the checkpoint from a "
            "local directory of .safetensors files (--model-path <checkpoint dir>); download it first, for example with "
            "huggingface-cli download <repo> --local-dir <dir>")
    return sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))


class SafetensorsCheckpoint:
    """A local directory of ``.safetensors`` files, loaded by every rank of a process group: each rank
    reads its share of the files into the OS page cache, applies each parameter's ``SafetensorsWeightLoader`` (shard,
    fuse, relayout) and uploads the result to the device."""

    def __init__(self, model_name_or_path: str):
        self._checkpoint_dir = model_name_or_path
        self._safetensor_file_names = _safetensors_files(model_name_or_path)
        assert len(self._safetensor_file_names) > 0, (
            f"No .safetensors files found for {model_name_or_path}"
        )

        # Populated lazily as files from checkpoint are loaded into OS page cache
        self._tensor_name_to_file = {}
        self._open_safetensor_files = {}

    def load_sharded_pipelined(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        mappings: dict,
        device: torch.device,
        dtype_override: dict[str, torch.dtype] | None = None,
    ) -> CheckpointLoadResult:
        """Load ``model``'s parameters for TP ``rank`` of ``world_size`` onto ``device``, each cast to its own dtype (or
        the one ``dtype_override`` names for it: parameters a cast model holds in a narrower dtype than they load in), in
        three overlapped stages: a background thread reads the files into the page cache; the main thread opens each
        file once it is cached and hands the parameters whose keys it holds to worker threads, which read the slices and
        apply the parameter's loader; a background thread uploads the results.

        ``mappings`` maps a parameter name to its checkpoint key, or to a list of keys for a fused parameter
        (``{"qkv_weight": ["q_proj.weight", "k_proj.weight", "v_proj.weight"]}``); an unmapped parameter loads the key
        of its own name. A key a mapping names that the checkpoint lacks raises ``RuntimeError``. Returns the
        ``CheckpointLoadResult``; the caller assigns it with ``model.load_state_dict(result.state_dict, strict=False,
        assign=True)``."""

        # Distributed key-value store for communicating which files have been loaded to OS page cache
        cached_files_store: torch.distributed.Store = (
            torch.distributed.distributed_c10d._get_default_store()
        )

        tensor_queue: queue.Queue = (
            queue.Queue()
        )  # Queue for pipelining file -> memory and memory -> HBM tensor transfer
        state_dict: dict[
            str, torch.Tensor
        ] = {}  # State dict populated with the rank's weights
        shutdown_event = (
            threading.Event()
        )  # Signal for clean shutdown if any thread has an error

        # One worker populates the page cache and one transfers tensors to the device; the remaining workers run the
        # per-parameter transforms (_process_param), each with one intra-op thread. torch.set_num_threads is process
        # global (a thread started later takes the last value set), so the caller's count is restored once the pool
        # is done.
        num_threads = torch.get_num_threads()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        futures: list = []
        named_params = list(model.named_parameters())
        unknown = set(dtype_override or {}) - {name for name, _ in named_params}
        if unknown:
            raise ValueError(f"dtype_override names no parameter of the model: {sorted(unknown)}")
        num_mapped_params = len(named_params)

        # Collect all checkpoint keys referenced by mappings for unexpected_keys calc
        referenced_checkpoint_keys: set[str] = set()
        for name, _ in named_params:
            ck = mappings.get(name, name)
            if isinstance(ck, list):
                referenced_checkpoint_keys.update(ck)
            else:
                referenced_checkpoint_keys.add(ck)

        try:
            # Start background threads for pipelined execution
            page_cache_future = executor.submit(
                self._load_to_page_cache,
                rank,
                world_size,
                cached_files_store,
                shutdown_event,
            )
            device_future = executor.submit(
                self._load_to_device,
                rank,
                tensor_queue,
                state_dict,
                device,
                num_mapped_params,
                shutdown_event,
            )
            futures.extend([page_cache_future, device_future])

            processed_files: set[str] = set()
            processed_param_names: set[str] = set()
            indexed_keys = -1  # the size of the tensor index at the last pass over the parameters

            while len(processed_files) < self.get_num_files():
                # Check if any new files have been read into cache
                for file_name in self._safetensor_file_names:
                    if file_name in processed_files:
                        continue
                    if cached_files_store.check([file_name]):
                        processed_files.add(file_name)

                        # Update tensor-to-file mapping for this file
                        file_path = os.path.join(self._checkpoint_dir, file_name)
                        self._open_safetensor_files[file_path] = safe_open(
                            file_path, framework="pt", device="cpu"
                        )
                        for key in self._open_safetensor_files[file_path].keys():
                            self._tensor_name_to_file[key] = file_path

                all_files_are_processed = len(processed_files) == self.get_num_files()

                # Process all parameters with weights that are now in cache; a parameter becomes ready only when the
                # tensor index grows, so the parameters are scanned again only then, and once all files are processed
                # (which refuses a parameter whose keys the checkpoint lacks)
                if len(self._tensor_name_to_file) == indexed_keys and not all_files_are_processed:
                    pending_params = ()
                else:
                    indexed_keys = len(self._tensor_name_to_file)
                    pending_params = named_params
                for name, param in pending_params:
                    if name in processed_param_names:
                        continue

                    # Get the checkpoint key names corresponding to parameter
                    # If not in mappings, use parameter name as checkpoint key
                    checkpoint_keys = mappings.get(name, name)
                    checkpoint_keys = (
                        checkpoint_keys
                        if isinstance(checkpoint_keys, list)
                        else [checkpoint_keys]
                    )

                    # Check if all weights needed to create the parameter are ready
                    if not all(
                        key in self._tensor_name_to_file for key in checkpoint_keys
                    ):
                        # All files from a checkpoint have been processed, but weight(s) are missing
                        if all_files_are_processed:
                            missing = [
                                k
                                for k in checkpoint_keys
                                if k not in self._tensor_name_to_file
                            ]
                            raise RuntimeError(
                                f"Checkpoint key(s) not found for parameter '{name}': {missing}"
                            )
                        continue

                    # Weights are ready to be processed
                    target_dtype = (dtype_override or {}).get(name, param.dtype)
                    weight_loader = get_weight_loader(param)
                    future = executor.submit(
                        self._process_param,
                        name,
                        checkpoint_keys,
                        weight_loader,
                        rank,
                        target_dtype,
                        tensor_queue,
                    )
                    futures.append(future)
                    processed_param_names.add(name)

                # Check if any futures raised exceptions (silent exceptions in futures could lead to hanging)
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception()

                if len(processed_files) < self.get_num_files():
                    time.sleep(_CACHED_FILE_POLL_S)

            # Wait for all futures to complete or for one to raise an exception
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            for future in done:
                # Raises an exception if there was one
                future.result()

        except Exception as e:
            logger.error(f"Exception during load_sharded_pipelined(): {e}")
            shutdown_event.set()
            tensor_queue.put(
                (None, None)
            )  # Unblock _load_to_device if waiting on queue
            raise e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            torch.set_num_threads(num_threads)

        unexpected_keys = [
            k for k in self._tensor_name_to_file if k not in referenced_checkpoint_keys
        ]

        return CheckpointLoadResult(state_dict, unexpected_keys)

    def _load_to_page_cache(
        self,
        rank: int,
        world_size: int,
        cached_files_store: torch.distributed.Store,
        shutdown_event: threading.Event,
    ) -> None:
        """Read this rank's share of the checkpoint files once, so they sit in the OS page cache: the shard reads that
        follow are strided, which is slow from disk and fast from memory. Rank r reads every file whose index i has i %
        world_size == r, and counts each file it cached in ``cached_files_store`` under the file's name (readers only
        check the key exists). ``shutdown_event`` stops it early when another stage failed."""
        import time

        start_time = time.perf_counter()

        # Used for logging the files read by this rank
        cached_files = []

        try:
            # Load all files into OS page cache
            for idx, file_name in enumerate(self._safetensor_file_names):
                if shutdown_event.is_set():
                    return

                if idx % world_size == rank:
                    file_path = os.path.join(self._checkpoint_dir, file_name)

                    # Read the file into the OS page cache with retries.
                    # A shutdown signal during a retry stops this rank early.
                    if not self._read_file_into_page_cache(file_path, shutdown_event):
                        return

                    # Add to distributed store so other threads/processes know this file is ready
                    cached_files_store.add(file_name, 1)
                    cached_files.append(file_name)

        except Exception as e:
            logger.error(f"Error in _load_to_page_cache(): {e}")
            raise

        logger.info(
            f"Finished populating page cache files {cached_files}: {time.perf_counter() - start_time}"
        )

    def _read_file_into_page_cache(
        self, file_path: str, shutdown_event: threading.Event
    ) -> bool:
        """Read ``file_path`` through once so it sits in the page cache, retrying with exponential backoff: a network
        file system (FSx, Lustre) can fail a read with an ``OSError`` such as a broken pipe, and a re-read is safe.
        False when ``shutdown_event`` stopped it during a backoff; ``OSError`` after the last attempt."""
        chunk_size = 4 * 1024 * 1024  # 4 MB
        backoff_s = _PAGE_CACHE_READ_BASE_BACKOFF_S
        for attempt in range(1, _PAGE_CACHE_READ_MAX_ATTEMPTS + 1):
            try:
                # Read through the file to load it into the OS page cache.
                with open(file_path, "rb") as f:
                    while f.read(chunk_size):
                        pass
                return True
            except OSError as e:
                if attempt >= _PAGE_CACHE_READ_MAX_ATTEMPTS:
                    logger.error(
                        "Failed to read file %s into the page cache after %d attempts: %s",
                        file_path,
                        attempt,
                        e,
                    )
                    raise
                logger.warning(
                    "Cannot read file %s into the page cache (attempt %d/%d): %s. "
                    "Retry in %.1f seconds.",
                    file_path,
                    attempt,
                    _PAGE_CACHE_READ_MAX_ATTEMPTS,
                    e,
                    backoff_s,
                )
                # Stop early if another thread signals a shutdown.
                if shutdown_event.wait(timeout=backoff_s):
                    return False
                backoff_s = min(_PAGE_CACHE_READ_MAX_BACKOFF_S, backoff_s * 2)

    def _process_param(
        self,
        name: str,
        checkpoint_keys: list[str],
        weight_loader: SafetensorsWeightLoader,
        rank: int,
        target_dtype: torch.dtype,
        tensor_queue: queue.Queue,
    ) -> None:
        """Read ``name``'s checkpoint slices (several keys for a fused parameter), apply its loader for ``rank``, cast
        to ``target_dtype`` and queue it for the upload thread."""
        torch.set_num_threads(1)  # one intra-op thread per transform worker (load_sharded_pipelined restores it)
        tensor = weight_loader.load([self._get_slice(k) for k in checkpoint_keys], rank)
        if tensor.dtype != target_dtype:
            logger.warning(
                f"Mismatch between parameter {name} defined dtype {target_dtype} and weight loader-generated "
                f"dtype {tensor.dtype}, casting to {target_dtype}."
            )
            tensor = tensor.to(target_dtype)
        tensor_queue.put((name, tensor))

    def _load_to_device(
        self,
        rank: int,
        tensor_queue: queue.Queue,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
        num_params: int,
        shutdown_event: threading.Event,
    ) -> None:
        """Move tensors from the host queue to device HBM as they arrive, while the other threads read, shard and pad
        the next ones. It blocks on an empty queue and returns after ``num_params`` tensors, or at the shutdown signal
        ``(None, None)``."""
        try:
            tensor_count = 0
            while tensor_count < num_params:
                if shutdown_event.is_set():
                    return

                name, cpu_tensor = tensor_queue.get()

                if name is None:  # Shutdown signal
                    return

                device_tensor = cpu_tensor.to(device)
                state_dict[name] = device_tensor
                tensor_count += 1

        except Exception as e:
            logger.error(f"[Rank {rank}] Error in _load_to_device(): {e}")
            raise

    def get_num_files(self) -> int:
        """The number of ``.safetensors`` files found in the checkpoint directory at construction."""
        return len(self._safetensor_file_names)

    def _get_slice(self, name: str) -> "PySafeSlice":
        """The lazy ``PySafeSlice`` of tensor ``name`` from the file that holds it, which reads only the parts indexed.
        """
        file_with_tensor = self._tensor_name_to_file[name]
        open_safetensor_file = self._open_safetensor_files[file_with_tensor]
        return open_safetensor_file.get_slice(name)

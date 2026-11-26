# SPDX-License-Identifier: Apache-2.0
"""Host-DRAM KV offloading (``--kv-offloading-size``) on Neuron.

``--kv-offloading-size`` selects vLLM's OffloadingConnector with its
CPUOffloadingSpec (or TieringOffloadingSpec, its subclass with disk / network
tiers behind the host tier). Their scheduler side (host block manager,
eviction, sizing, secondary tiers) is device-neutral and runs unchanged. Two
worker-side pieces are not:

- ``OffloadingConnectorWorker.register_kv_caches`` views each layer as one
  blocks-first region. Neuron allocates a layer as
  (2, num_blocks, num_kv_heads, block_size, head_size): K and V are separate
  num_blocks-sized slabs, so one block is two non-contiguous byte ranges. As
  NeuronNixlConnector does for NIXL, ``register_kv_caches`` registers each
  slab as its own blocks-first region.
- ``CPUOffloadingWorker`` copies with CUDA streams and kernels.
  ``NeuronCPUOffloadingWorker`` copies runs of whole blocks between each slab
  and its host pages with ``Tensor.copy_``, which the Neuron runtime runs
  eagerly (torch on the host in CPU mode). A copy has landed when it returns,
  so each job completes inside ``submit_*``.
"""

import time

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def register_kv_caches(connector, kv_caches: dict) -> None:
    """Register the runner's KV caches with ``connector``.

    ``kv_caches`` maps each layer to its (num_caches, num_blocks, ...) tensor
    or [K, V] list; iterating either yields the blocks-first slabs. Layers
    that share a buffer share its regions.
    """
    if not isinstance(connector, OffloadingConnector):
        connector.register_kv_caches(kv_caches)
        return
    worker = connector.connector_worker
    assert worker is not None
    spec = worker.spec
    if not isinstance(spec, CPUOffloadingSpec) or spec.config.canonical_layout:
        raise NotImplementedError(
            "Neuron KV offloading supports CPUOffloadingSpec and "
            f"TieringOffloadingSpec without canonical_layout, not {type(spec).__name__}"
        )
    tensors: list[CanonicalKVCacheTensor] = []
    tensor_idx: dict[int, int] = {}
    group_refs: list[list[CanonicalKVCacheRef]] = []
    for group in worker.kv_cache_config.kv_cache_groups:
        refs: dict[int, CanonicalKVCacheRef] = {}
        for layer_name in group.layer_names:
            for slab in kv_caches[layer_name]:
                idx = tensor_idx.setdefault(slab.data_ptr(), len(tensors))
                if idx == len(tensors):
                    page = slab[0].numel() * slab.element_size()
                    tensors.append(CanonicalKVCacheTensor(slab, page))
                refs[idx] = CanonicalKVCacheRef(idx, tensors[idx].page_size_bytes)
        group_refs.append(list(refs.values()))
    region = None
    if isinstance(spec, TieringOffloadingSpec):
        # The secondary tiers (scheduler process) read and write the host
        # blocks through this file-backed region, as with CPUOffloadingWorker.
        region = SharedOffloadRegion(
            engine_id=spec.config.engine_id,
            num_blocks=spec.num_blocks,
            rank=0 if spec.replicated_layout else spec.config.parallel.rank,
            kv_bytes_per_block=spec.kv_bytes_per_chunk,
            cpu_page_size=spec.cpu_page_size_per_worker,
        )
    worker.worker = NeuronCPUOffloadingWorker(
        CanonicalKVCaches(tensors, group_refs),
        spec.blocks_per_chunk,
        spec.num_blocks,
        region,
    )


def drain_before_transfers(connector, metadata, prev_async_output) -> None:
    """Read back the previous step's async output before this step's copies.

    The copies read KV the previous step's graph wrote (stores) or write
    blocks it may still read (loads). A graph leaves the Neuron execution
    queue only when its output is read back, and with TP > 1 a copy of its KV
    issued before that blocks forever. ``get_output`` is idempotent, so the
    async-output thread then finds it done.
    """
    if prev_async_output is None or not isinstance(connector, OffloadingConnector):
        return
    worker = connector.connector_worker
    if worker._unsubmitted_store_jobs or metadata.jobs_to_flush or metadata.load_jobs:
        prev_async_output.get_output()


def block_runs(device_blocks, host_pages, blocks_per_chunk: int) -> list[list[int]]:
    """[device block, host page, count] runs of pairs consecutive on both
    sides. With blocks_per_chunk > 1 a run stays inside one host block."""
    runs: list[list[int]] = []
    for d, h in zip(device_blocks, host_pages):
        if (
            runs
            and runs[-1][0] + runs[-1][2] == d
            and runs[-1][1] + runs[-1][2] == h
            and (blocks_per_chunk == 1 or h % blocks_per_chunk)
        ):
            runs[-1][2] += 1
        else:
            runs.append([d, h, 1])
    return runs


class NeuronCPUOffloadingWorker(OffloadingWorker):
    """Copies KV blocks between the device slabs and host memory.

    Host block b of tensor i holds ``blocks_per_chunk`` device-sized pages of
    tensor i, in the byte layout CPUOffloadingWorker uses: a private
    (num_cpu_blocks, page * blocks_per_chunk) tensor, or the tensor's view of
    ``region``. Host memory is not pinned: the Neuron device has no
    pinned-memory allocator.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        region: SharedOffloadRegion | None = None,
    ):
        self.region = region
        self.blocks_per_chunk = blocks_per_chunk
        self.device_tensors = [t.tensor for t in kv_caches.tensors]
        self.host_tensors = []
        for t in kv_caches.tensors:
            nbytes = t.page_size_bytes * blocks_per_chunk
            rows = (
                region.create_next_worker_view(nbytes)
                if region is not None
                else torch.zeros((num_cpu_blocks, nbytes), dtype=torch.int8)
            )
            block_shape = (blocks_per_chunk, *t.tensor.shape[1:])
            self.host_tensors.append(rows.view(t.tensor.dtype).unflatten(1, block_shape))
        self.group_tensor_ids = [
            [ref.tensor_idx for ref in refs] for refs in kv_caches.group_data_refs
        ]
        self._finished: list[TransferResult] = []

    def _host_pages(self, t: int, h: int, n: int) -> torch.Tensor:
        host = self.host_tensors[t]
        if self.blocks_per_chunk == 1:
            return host[h : h + n, 0]
        row, sub = divmod(h, self.blocks_per_chunk)
        return host[row, sub : sub + n]

    def _transfer(
        self,
        job_id: int,
        device_spec: GPULoadStoreSpec,
        host_spec: LoadStoreSpec,
        to_host: bool,
    ) -> bool:
        start = time.perf_counter()
        on_device = self.device_tensors[0].device.type != "cpu"
        if on_device:
            # stores read KV the launched graphs write; loads overwrite blocks
            # they may still read
            torch.neuron.synchronize()
        bpc = self.blocks_per_chunk
        device_blocks, host_blocks = device_spec.block_ids, host_spec.block_ids
        num_bytes = dev_off = host_off = 0
        for tensor_ids, size, index in zip(
            self.group_tensor_ids, device_spec.group_sizes, device_spec.block_indices
        ):
            if size == 0:
                continue
            # the group's first block may start inside its first host block
            skip = index % bpc
            num_host = cdiv(size + skip, bpc)
            pages = host_blocks[host_off : host_off + num_host, None] * bpc + np.arange(bpc)
            runs = block_runs(
                device_blocks[dev_off : dev_off + size].tolist(),
                pages.ravel()[skip : skip + size].tolist(),
                bpc,
            )
            for t in tensor_ids:
                dev = self.device_tensors[t]
                for d, h, n in runs:
                    # the Neuron runtime copies only contiguous host memory
                    # correctly, so strided region pages go through a copy
                    host = self._host_pages(t, h, n)
                    if to_host:
                        host.copy_(dev[d : d + n] if host.is_contiguous() else dev[d : d + n].cpu())
                    else:
                        dev[d : d + n].copy_(host.contiguous())
                num_bytes += size * self.host_tensors[t][0, 0].nbytes
            dev_off += size
            host_off += num_host
        assert dev_off == len(device_blocks) and host_off == len(host_blocks)
        if on_device and not to_host:
            torch.neuron.synchronize()  # the next forward reads the loaded blocks
        self._finished.append(
            TransferResult(job_id, True, num_bytes, time.perf_counter() - start)
        )
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._transfer(job_id, src_spec, dst_spec, to_host=True)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._transfer(job_id, dst_spec, src_spec, to_host=False)

    def get_finished(self) -> list[TransferResult]:
        finished, self._finished = self._finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        pass  # jobs complete in submit_*

    def shutdown(self) -> None:
        self.host_tensors.clear()
        if self.region is not None:
            self.region.cleanup()
            self.region = None

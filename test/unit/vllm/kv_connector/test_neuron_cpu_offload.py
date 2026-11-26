# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Neuron host-DRAM KV offloading (neuron_cpu_offload.py).

Host tensors stand in for the Neuron KV slabs (CPU mode). Covers the K/V-slab
registration, the block-run math, store/load round trips (offloaded blocks
larger than device blocks, several KV groups) and the shared-region layout
the tiering spec's secondary tiers read.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest test/unit/vllm/kv_connector/test_neuron_cpu_offload.py -v
"""

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from vllm_neuron.vllm.kv_connector import neuron_cpu_offload as nco

NUM_BLOCKS, HEADS, BLOCK, HEAD_DIM = 16, 2, 4, 256  # one page = 4 KiB, region-aligned


def _layer():
    """A Neuron layer cache: (2, num_blocks, heads, block, head_dim), K then V."""
    return torch.randn(2, NUM_BLOCKS, HEADS, BLOCK, HEAD_DIM, dtype=torch.bfloat16)


def _caches(layers):
    tensors, refs = [], []
    for full in layers:
        for slab in full:
            page = slab[0].numel() * slab.element_size()
            refs.append(CanonicalKVCacheRef(len(tensors), page))
            tensors.append(CanonicalKVCacheTensor(slab, page))
    return CanonicalKVCaches(tensors, [refs])


def _connector(spec, groups):
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = SimpleNamespace(
        spec=spec,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=g) for g in groups]
        ),
        worker=None,
    )
    return connector


def _spec(cls, blocks_per_chunk=1, num_blocks=8, **attrs):
    spec = object.__new__(cls)
    spec.config = SimpleNamespace(
        canonical_layout=False, engine_id=uuid.uuid4().hex,
        parallel=SimpleNamespace(rank=0),
    )
    spec.blocks_per_chunk = blocks_per_chunk
    spec.num_blocks = num_blocks
    spec.replicated_layout = False
    for k, v in attrs.items():
        setattr(spec, k, v)
    return spec


def test_block_runs_merges_pairs_consecutive_on_both_sides():
    assert nco.block_runs([3, 4, 5, 9], [0, 1, 2, 3], 1) == [[3, 0, 3], [9, 3, 1]]
    assert nco.block_runs([3, 4, 6], [5, 6, 7], 1) == [[3, 5, 2], [6, 7, 1]]


def test_block_runs_stop_at_host_block_boundary():
    # pages 2,3 | 4,5 with 2 pages per host block: two runs despite contiguity
    assert nco.block_runs([0, 1, 2, 3], [2, 3, 4, 5], 2) == [[0, 2, 2], [2, 4, 2]]


def test_register_splits_each_layer_into_k_and_v_regions():
    layers = {"l0": _layer(), "l1": _layer()}
    connector = _connector(_spec(CPUOffloadingSpec), [["l0", "l1"]])
    nco.register_kv_caches(connector, layers)
    worker = connector.connector_worker.worker
    assert isinstance(worker, nco.NeuronCPUOffloadingWorker)
    assert [t.data_ptr() for t in worker.device_tensors] == [
        layers["l0"][0].data_ptr(), layers["l0"][1].data_ptr(),
        layers["l1"][0].data_ptr(), layers["l1"][1].data_ptr(),
    ]
    assert worker.group_tensor_ids == [[0, 1, 2, 3]]
    assert tuple(worker.host_tensors[0].shape) == (8, 1, HEADS, BLOCK, HEAD_DIM)


def test_register_shares_regions_of_layers_on_one_buffer():
    shared = _layer()
    connector = _connector(_spec(CPUOffloadingSpec), [["a", "b"]])
    nco.register_kv_caches(connector, {"a": shared, "b": [shared[0], shared[1]]})
    worker = connector.connector_worker.worker
    assert len(worker.device_tensors) == 2
    assert worker.group_tensor_ids == [[0, 1]]


def test_register_passes_other_connectors_through():
    calls = []
    other = SimpleNamespace(register_kv_caches=calls.append)
    caches = {"l0": _layer()}
    nco.register_kv_caches(other, caches)
    assert calls == [caches]


def test_register_rejects_canonical_layout():
    spec = _spec(CPUOffloadingSpec)
    spec.config.canonical_layout = True
    with pytest.raises(NotImplementedError):
        nco.register_kv_caches(_connector(spec, [["l0"]]), {"l0": _layer()})


def test_store_then_load_round_trip():
    layers = [_layer(), _layer()]
    worker = nco.NeuronCPUOffloadingWorker(_caches(layers), 1, 8)
    original = [full.clone() for full in layers]
    blocks = [2, 3, 4, 9]
    assert worker.submit_store(1, GPULoadStoreSpec(blocks, [4], [0]), CPULoadStoreSpec([5, 6, 7, 0]))
    for full in layers:
        full.zero_()
    assert worker.submit_load(2, CPULoadStoreSpec([5, 6, 7, 0]), GPULoadStoreSpec([10, 11, 12, 1], [4], [0]))
    for full, ref in zip(layers, original):
        # K and V of each block land in the new blocks, nothing else is written
        assert torch.equal(full[:, [10, 11, 12, 1]], ref[:, blocks])
        assert not full[:, [0, 2, 3, 4, 5]].any()
    results = worker.get_finished()
    page = HEADS * BLOCK * HEAD_DIM * 2
    assert [(r.job_id, r.success, r.transfer_size) for r in results] == [
        (1, True, 4 * 4 * page), (2, True, 4 * 4 * page),
    ]
    assert worker.get_finished() == []


def test_load_with_larger_host_blocks_and_unaligned_start():
    # 2 device blocks per host block; the request's blocks 3..6 start halfway
    # into host block 7 (logical block 3 = second page of chunk 1)
    full = _layer()
    worker = nco.NeuronCPUOffloadingWorker(_caches([full]), 2, 8)
    ref = full.clone()
    worker.submit_store(1, GPULoadStoreSpec([0, 1, 2, 3, 4], [5], [1]), CPULoadStoreSpec([7, 2, 5]))
    host = worker.host_tensors[0]
    assert torch.equal(host[7, 1], ref[0, 0])
    assert torch.equal(host[2, 0], ref[0, 1]) and torch.equal(host[2, 1], ref[0, 2])
    assert torch.equal(host[5, 1], ref[0, 4])
    full.zero_()
    worker.submit_load(2, CPULoadStoreSpec([2, 5]), GPULoadStoreSpec([8, 9, 10], [3], [2]))
    assert torch.equal(full[:, [8, 9, 10]], ref[:, [1, 2, 3]])


def test_multiple_groups_use_their_own_tensors():
    a, b = _layer(), _layer()
    caches = _caches([a, b])
    caches = CanonicalKVCaches(caches.tensors, [caches.group_data_refs[0][:2], caches.group_data_refs[0][2:]])
    worker = nco.NeuronCPUOffloadingWorker(caches, 1, 8)
    ref_a, ref_b = a.clone(), b.clone()
    worker.submit_store(1, GPULoadStoreSpec([1, 2, 7], [2, 1], [0, 0]), CPULoadStoreSpec([0, 1, 2]))
    a.zero_()
    b.zero_()
    worker.submit_load(2, CPULoadStoreSpec([0, 1, 2]), GPULoadStoreSpec([3, 4, 5], [2, 1], [0, 0]))
    assert torch.equal(a[:, [3, 4]], ref_a[:, [1, 2]]) and not a[:, 5].any()
    assert torch.equal(b[:, 5], ref_b[:, 7]) and not b[:, [3, 4]].any()


def test_tiering_spec_uses_the_shared_region_layout():
    """Host blocks live in the region in CPUOffloadingWorker's byte layout:
    row b = [K0 | V0 | K1 | V1] pages of block b, which the secondary tiers
    copy whole."""
    layers = {"l0": _layer(), "l1": _layer()}
    page = HEADS * BLOCK * HEAD_DIM * 2
    spec = _spec(TieringOffloadingSpec, kv_bytes_per_chunk=4 * page, cpu_page_size_per_worker=4 * page)
    connector = _connector(spec, [["l0", "l1"]])
    nco.register_kv_caches(connector, layers)
    worker = connector.connector_worker.worker
    try:
        worker.submit_store(1, GPULoadStoreSpec([6, 7], [2], [0]), CPULoadStoreSpec([3, 4]))
        rows = np.frombuffer(worker.region.create_kv_memoryview(), dtype=np.uint8).reshape(8, 4 * page)
        for i, slab in enumerate([layers["l0"][0], layers["l0"][1], layers["l1"][0], layers["l1"][1]]):
            for host_block, dev_block in ((3, 6), (4, 7)):
                expect = slab[dev_block].contiguous().view(torch.uint8).numpy().ravel()
                assert np.array_equal(rows[host_block, i * page : (i + 1) * page], expect)
        refs = {name: full.clone() for name, full in layers.items()}
        for full in layers.values():
            full.zero_()
        worker.submit_load(2, CPULoadStoreSpec([4, 3]), GPULoadStoreSpec([0, 1], [2], [0]))
        for name, full in layers.items():
            assert torch.equal(full[:, [0, 1]], refs[name][:, [7, 6]])
            assert not full[:, 2:].any()
    finally:
        worker.shutdown()


def test_region_cleanup_on_shutdown():
    spec = _spec(TieringOffloadingSpec, kv_bytes_per_chunk=4096, cpu_page_size_per_worker=4096)
    region = SharedOffloadRegion(spec.config.engine_id, 2, 0, 4096, 4096)
    worker = nco.NeuronCPUOffloadingWorker(CanonicalKVCaches([], []), 1, 2, region)
    worker.shutdown()
    assert worker.region is None


def test_drain_reads_back_previous_step_only_before_offload_copies():
    reads = []
    prev = SimpleNamespace(get_output=lambda: reads.append(1))
    connector = _connector(_spec(CPUOffloadingSpec), [["l0"]])
    connector.connector_worker._unsubmitted_store_jobs = []
    idle = SimpleNamespace(jobs_to_flush=set(), load_jobs={})
    nco.drain_before_transfers(connector, idle, prev)
    assert reads == []
    nco.drain_before_transfers(connector, SimpleNamespace(jobs_to_flush=set(), load_jobs={7: None}), prev)
    assert reads == [1]
    nco.drain_before_transfers(connector, SimpleNamespace(jobs_to_flush={3}, load_jobs={}), prev)
    assert reads == [1, 1]
    connector.connector_worker._unsubmitted_store_jobs = [(5, None, None)]
    nco.drain_before_transfers(connector, idle, prev)
    assert reads == [1, 1, 1]
    nco.drain_before_transfers(connector, idle, None)  # first step: nothing to drain
    nco.drain_before_transfers(SimpleNamespace(), idle, prev)  # other connectors
    assert reads == [1, 1, 1]

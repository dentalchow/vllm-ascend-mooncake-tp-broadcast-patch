#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""TP-replicated KV cache broadcast for the AscendStore (Mooncake) KV pool.

Background
----------
For attention types whose KV cache is byte-identical on every TP rank
(e.g. the MLA/DSA latent cache used by DeepSeek-V3/V4), there is no need
for every TP rank to independently issue a remote read against the
external KV store (Mooncake): TP rank 0 can fetch the one copy that
exists remotely and fan it out to the other TP ranks over the
intra-node HCCS interconnect, which is both much faster than another
round of RDMA reads and removes (tp_size - 1) redundant RDMA NIC users
per request.

This module intentionally does not depend on exactly how
`kv_transfer.py`'s request-handling classes are shaped (that file
changes often); it only depends on:
  - a TP rank/size,
  - a list of raw NPU device addresses + byte sizes that the caller has
    already resolved locally (the KV cache blocks live at different
    physical addresses on every rank, but at the same logical offset),
  - the existing `vllm_ascend.distributed.device_communicators.pyhccl`
    raw-pointer HCCL wrapper, which is the same primitive
    `distributed/weight_transfer/hccl_engine.py` uses for weight
    broadcasting.

Caveats (read before enabling in production)
---------------------------------------------
- Only correct for KV cache groups whose blocks are guaranteed
  byte-identical on every TP rank. This holds for MLA/DSA-style latent
  caches (DeepSeek V2/V3/V4, etc.) when TP > 1 *without* DCP/PCP sharding
  of that same group. It is NOT correct for ordinary GQA/MHA KV heads
  sharded across TP ranks, or for hybrid models where only some KV
  groups are replicated (e.g. Kimi-Linear style models that mix MLA with
  sharded recurrent state) -- enabling this for the wrong group silently
  serves rank 0's shard to every rank, which is a correctness bug, not a
  crash. There is no automatic signal for this in this vllm-ascend
  revision, so group selection is explicit and operator-controlled (see
  `vllm_ascend/envs.py`'s `VLLM_ASCEND_KV_TP_BROADCAST_GROUPS`).
- Not validated on real Ascend hardware by this patch; validate the
  broadcast path (compare broadcast-loaded bytes against a plain
  per-rank remote read) before relying on it in production.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterable, Sequence

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    aclrtStream_t,
    buffer_type,
    hcclDataTypeEnum,
)
from vllm_ascend.utils import current_stream

# Default per-call broadcast shard size (bytes). A single large block
# buffer is split into shards of at most this size so that:
#  - one hcclBroadcast call cannot monopolize the HCCS ring for an
#    unbounded amount of time, and
#  - shards are issued back-to-back on a dedicated NPU stream, which
#    lets the NPU pipeline issuing the next shard's collective with the
#    previous shard's data movement instead of fully serializing them
#    from the host/Python side.
DEFAULT_SHARD_BYTES = 8 * 1024 * 1024


class _AllGroups:
    """Sentinel: every kv_cache_group_id is considered TP-replicated."""

    def __contains__(self, item: object) -> bool:
        return True

    def __repr__(self) -> str:
        return "<all kv_cache groups>"


ALL_GROUPS = _AllGroups()


def parse_broadcast_group_ids(raw: str | None) -> frozenset[int] | _AllGroups:
    """Parse `VLLM_ASCEND_KV_TP_BROADCAST_GROUPS`.

    - unset / empty string -> no groups (feature disabled)
    - "all"                -> every kv_cache_group_id (`ALL_GROUPS`)
    - "0,2"                -> only kv_cache_group_ids 0 and 2
    """
    if not raw:
        return frozenset()
    raw = raw.strip()
    if raw.lower() == "all":
        return ALL_GROUPS
    try:
        return frozenset(int(part) for part in raw.split(",") if part.strip() != "")
    except ValueError:
        logger.warning(
            "[KVPool][TPBroadcast] Invalid VLLM_ASCEND_KV_TP_BROADCAST_GROUPS=%r "
            "(expected 'all' or a comma-separated list of integers); "
            "TP broadcast is disabled.",
            raw,
        )
        return frozenset()


class TpKvBroadcaster:
    """Lazily-initialized HCCL broadcaster bound to the TP process group.

    One instance is owned by each `KVCacheStoreRecvingThread`. All calls
    in this class are collective: every TP rank must call the same
    method, in the same order, with the same number/order of entries,
    for the same logical request -- this already holds naturally here
    because every TP rank's connector receives identical per-step
    connector metadata (same requests, same block ids, same iteration
    order) from the scheduler; only the raw addresses differ (each
    rank's KV cache tensors are separate physical allocations at the
    same logical block offset).
    """

    def __init__(self, tp_rank: int, tp_size: int, shard_bytes: int = DEFAULT_SHARD_BYTES):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.shard_bytes = max(1, int(shard_bytes))
        self._comm: PyHcclCommunicator | None = None
        self._stream: torch.npu.Stream | None = None
        self._status_buf: torch.Tensor | None = None
        self._init_lock = threading.Lock()
        self._init_failed = False

    @property
    def enabled(self) -> bool:
        return self.tp_size > 1 and not self._init_failed

    def _ensure_init(self) -> bool:
        """Returns True if a usable HCCL communicator is ready."""
        if self._comm is not None:
            return self._comm.available
        if not self.enabled:
            return False
        with self._init_lock:
            if self._comm is not None:
                return self._comm.available
            try:
                tp_group = get_tp_group()
                device = torch.npu.current_device()
                comm = PyHcclCommunicator(tp_group.cpu_group, device=device)
            except Exception:
                logger.exception(
                    "[KVPool][TPBroadcast] Failed to initialize the TP KV "
                    "broadcast HCCL communicator; falling back to per-rank "
                    "remote reads for broadcast-eligible groups."
                )
                self._init_failed = True
                return False
            if not comm.available:
                logger.warning(
                    "[KVPool][TPBroadcast] PyHcclCommunicator reports "
                    "unavailable (missing HCCL library?); falling back to "
                    "per-rank remote reads for broadcast-eligible groups."
                )
                self._comm = comm
                self._init_failed = True
                return False
            self._stream = torch.npu.Stream(device=device)
            self._status_buf = torch.zeros(1, dtype=torch.uint8, device=device)
            self._comm = comm
            logger.info(
                "[KVPool][TPBroadcast] TP KV broadcast communicator ready "
                "(tp_rank=%d tp_size=%d shard_bytes=%d)",
                self.tp_rank,
                self.tp_size,
                self.shard_bytes,
            )
            return True

    def _iter_shards(self, addrs: Sequence[int], sizes: Sequence[int]) -> Iterable[tuple[int, int]]:
        for addr, size in zip(addrs, sizes):
            if size <= 0:
                continue
            offset = 0
            while offset < size:
                chunk = min(self.shard_bytes, size - offset)
                yield addr + offset, chunk
                offset += chunk

    def broadcast_entries(self, addr_list: Sequence[int], size_list: Sequence[int], src: int = 0) -> bool:
        """Broadcast `[addr, addr + size)` byte ranges from `src` to every
        TP rank (including `src`, which is a no-op copy for itself).

        `addr_list`/`size_list` must be the *flattened* per-entry address
        and size lists, computed locally on this rank, for the exact same
        ordered set of (key, layer) entries on every rank.

        Returns True if the broadcast was issued and the stream was
        synchronized; False if the HCCL communicator is unavailable, in
        which case the caller MUST fall back to each rank doing its own
        remote read for correctness.
        """
        if not addr_list:
            return self.enabled and self._ensure_init()
        if not self._ensure_init():
            return False
        assert self._comm is not None
        stream = self._stream or current_stream()
        for addr, size in self._iter_shards(addr_list, size_list):
            self._comm.hccl.hcclBroadcast(
                buffer_type(addr),
                size,
                hcclDataTypeEnum.hcclUint8,
                src,
                self._comm.comm,
                aclrtStream_t(stream.npu_stream),
            )
        stream.synchronize()
        return True

    def broadcast_nested(
        self,
        addr_list: Sequence[Sequence[int]],
        size_list: Sequence[Sequence[int]],
        src: int = 0,
    ) -> bool:
        """Same as `broadcast_entries` but accepts the nested
        `list[list[int]]` shape `kv_transfer.py` uses (one outer entry per
        key, one inner entry per physical buffer/layer for that key)."""
        flat_addr = [a for sub in addr_list for a in sub]
        flat_size = [s for sub in size_list for s in sub]
        return self.broadcast_entries(flat_addr, flat_size, src=src)

    def broadcast_status(self, ok: bool, src: int = 0) -> bool:
        """Broadcast a single ok/fail flag from `src` to every rank.

        Used so that every TP rank -- not just `src`, which is the only
        rank that actually called the remote store -- can agree on
        whether the data that was just broadcast is valid, and fall back
        to recompute together instead of silently diverging.
        """
        if not self._ensure_init():
            return ok
        assert self._comm is not None and self._status_buf is not None
        stream = self._stream or current_stream()
        if self.tp_rank == src:
            self._status_buf.fill_(1 if ok else 0)
        self._comm.hccl.hcclBroadcast(
            buffer_type(self._status_buf.data_ptr()),
            1,
            hcclDataTypeEnum.hcclUint8,
            src,
            self._comm.comm,
            aclrtStream_t(stream.npu_stream),
        )
        stream.synchronize()
        return bool(self._status_buf.item())

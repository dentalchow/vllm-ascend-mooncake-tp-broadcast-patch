#!/usr/bin/env python3
"""Standalone HCCS point-to-point bandwidth probe between two *specific*
physical NPUs on the same node (e.g. npu1 and npu4).

This does NOT depend on the TP-broadcast KV patch at all -- it is a
pre-flight check you should run BEFORE investing in a cross-DP-group
"read-proxy" relay design. It answers one question: does an HCCL
broadcast between these two physical chips actually run at HCCS speed
(hundreds of GB/s), or does it fall back to something much slower
(e.g. because the two chips are on different boards / not on the same
HCCS switch domain, or because of some other topology constraint)?

It reuses vllm-ascend's own `PyHcclCommunicator` (same code path the
TP-broadcast patch uses), so the number you get here is representative
of what the patch would actually achieve.

Usage (run on the SAME host, two separate terminals/processes):

    # terminal A -- bind to physical npu1, acts as the broadcast root
    python3 hccs_p2p_bandwidth_probe.py --rank 0 --device 1 \
        --master-addr 127.0.0.1 --master-port 29500

    # terminal B -- bind to physical npu4, receives the broadcast
    python3 hccs_p2p_bandwidth_probe.py --rank 1 --device 4 \
        --master-addr 127.0.0.1 --master-port 29500

Both processes must be started within a few seconds of each other
(they rendezvous via a TCP store). Swap --device values to probe any
other pair, e.g. npu0<->npu1 (expected to be fast, same board) as a
sanity-check baseline to compare against npu1<->npu4.

Requires: torch, torch_npu, and vllm_ascend importable in this
environment (run it from inside your vllm-ascend conda/venv on the
NPU server).
"""

import argparse
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True, choices=[0, 1],
                         help="rank within this 2-process probe (0 = broadcast root)")
    parser.add_argument("--device", type=int, required=True,
                         help="physical NPU device id this process binds to, e.g. 1 or 4")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--size-mb", type=int, default=1024,
                         help="payload size per broadcast, in MiB (default 1024 = 1GiB, "
                              "matching your ~1GB per-request KV cache load)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    import torch_npu  # noqa: F401  (registers the npu backend / torch.npu.*)
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    torch.npu.set_device(args.device)

    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    dist.init_process_group(backend="gloo", init_method=init_method, world_size=2, rank=args.rank)

    comm = PyHcclCommunicator(group=dist.group.WORLD, device=args.device)
    if comm.disabled:
        print("PyHcclCommunicator failed to initialize (missing HCCL library or "
              "non-NPU environment) -- cannot probe bandwidth here.")
        dist.destroy_process_group()
        return

    numel = (args.size_mb * 1024 * 1024) // 4  # float32 elements
    buf = torch.empty(numel, dtype=torch.float32, device=f"npu:{args.device}")
    if args.rank == 0:
        buf.fill_(1.0)

    stream = torch.npu.current_stream()

    for _ in range(args.warmup):
        comm.broadcast(buf, src=0, stream=stream)
    stream.synchronize()

    elapsed_list = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        comm.broadcast(buf, src=0, stream=stream)
        stream.synchronize()
        elapsed_list.append(time.perf_counter() - t0)

    total_bytes = numel * 4
    gbps = [total_bytes / e / (1024 ** 3) for e in elapsed_list]
    avg_gbps = sum(gbps) / len(gbps)
    min_gbps = min(gbps)
    max_gbps = max(gbps)

    print(
        f"[rank={args.rank} device=npu{args.device}] payload={args.size_mb}MiB "
        f"iters={args.iters} avg={avg_gbps:.1f}GB/s min={min_gbps:.1f}GB/s "
        f"max={max_gbps:.1f}GB/s"
    )
    print(
        "Interpretation: on-board HCCS links typically sustain on the order of "
        "100+ GB/s. If this number is instead in the single-digit-GB/s range or "
        "close to your RDMA NIC's line rate, these two chips are very likely NOT "
        "on the same HCCS full-mesh domain, and the 'read-proxy + HCCS relay' "
        "design will not give you the speedup you're hoping for -- it would just "
        "move the bottleneck from the RDMA NIC to this NPU-NPU link instead."
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Mooncake(ascend_store) KV 池 TP 广播优化补丁

针对 vllm-ascend commit `50e0ce60849a7b6fe6f94`（tag `v0.25.1rc1`）+
vllm commit `752a3a504485790a2`（tag `v0.25.1`）的固定版本手写的补丁，
为 `AscendStore`（Mooncake）KV 池的接收路径实现「rank0 读远端 + HCCS 广播」，
思路对齐 LMCache-Ascend 对 MLA 类 KV（如 DeepSeek V3/V4 的 latent cache）
已经做的同类优化。

## 这个补丁做了什么

- **新增文件** `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/tp_kv_broadcast.py`
  - `TpKvBroadcaster`：懒加载的 HCCL 通信器（复用
    `distributed/device_communicators/pyhccl.py` 的 `PyHcclCommunicator`，
    和权重广播用的是同一套底层 ctypes 封装），直接对裸设备指针
    （`addr`/`size`，int 类型）做 `hcclBroadcast`，不需要也不可能把这些
    指针包装成 `torch.Tensor`。
  - 按 `VLLM_ASCEND_KV_TP_BROADCAST_SHARD_BYTES`（默认 8MiB）分片广播，
    在一条专用 NPU stream 上连续下发多个分片的 `hcclBroadcast`，
    只在最后统一 `synchronize()` 一次——这就是你要的"流水线"效果：
    下一个分片的下发不需要等上一个分片真正传输完。
  - 零拷贝：广播的目的地址就是本 rank 自己 KV cache 的物理地址，
    和 LMCache-Ascend 用「暂存 buffer + scatter」的方式不同，
    不需要额外的 load stream 和二次搬运。

- **修改** `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
  - `KVCacheStoreRecvingThread.__init__`：读取两个新环境变量，决定哪些
    `kv_cache_group_id` 可以走广播路径，并懒构造 `TpKvBroadcaster`。
  - `_handle_request`：把每个请求的 key/addr/size/block_id 按
    group_id 分成「普通组」（行为完全不变，仍然是每个 rank 各自按
    `tp_rank` 轮转读取自己那一份 key）和「广播组」两部分。
  - 新增 `_handle_broadcast_group_entries`：广播组里只有 `tp_rank==0`
    调用 `m_store.get()`，然后所有 rank（包括 rank0）一起做一次
    `broadcast_nested()`；再广播一个 1 字节的"整体成功/失败"状态，
    所有 rank 用它来判断这批 block 是否要标记为失效（宁可整体重算，
    也不去猜测具体哪个 key 失败——如果需要更精细的按 key 容错，
    需要额外给 Mooncake 的 `ret` 码本身也做一次广播，复杂度和收益
    现在不成正比，所以先按"整体成败"处理）。
  - **容错**：如果 HCCL 通信器初始化失败（比如缺库），非 root rank
    会自动退回到和补丁前完全一样的"自己读一份"逻辑，绝不会因为广播
    不可用就悄悄读到错误数据。

- **修改** `vllm_ascend/envs.py`：新增两个环境变量的声明/文档（见下）。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_ASCEND_KV_TP_BROADCAST_GROUPS` | `""`（关闭） | 哪些 `kv_cache_group_id` 的内容在所有 TP rank 上是逐字节相同的，可以走广播。`"all"` 表示全部组；`"0,2"` 表示只有 0 和 2 号组。**你的场景（DeepSeek V4 Flash 的 MLA KV）通常只有一个组，设成 `all` 即可。** |
| `VLLM_ASCEND_KV_TP_BROADCAST_SHARD_BYTES` | `8388608`（8MiB） | 每次 `hcclBroadcast` 调用搬运的最大字节数，用来做流水线分片。 |
| `VLLM_ASCEND_KV_TP_BROADCAST_ROOT_RANK` | `0` | **哪个 local TP rank**（0..tp_size-1，仅限本 DP 副本自己的 TP 组内编号）负责真正调用 `m_store.get()` 读远端存储，其余 rank 只接收 HCCS 广播。多 DP 副本、RDMA 亲和卡不统一时需要单独配置，见下一节。 |

`GROUPS` / `SHARD_BYTES` 需要在**每一个 TP worker 进程**启动前设置好，
且必须所有 rank 保持一致，否则会出现有的 rank 以为该走广播、有的 rank
以为不走，导致请求处理方式不一致甚至挂起。`ROOT_RANK` **可以、且在你的
拓扑下必须**按进程单独设置（见下一节），同一个 TP 组内的所有 rank 必须
算出同一个 `ROOT_RANK` 的值，否则没人读/两个人读会直接出错或死锁。

```bash
export VLLM_ASCEND_KV_TP_BROADCAST_GROUPS=all
export VLLM_ASCEND_KV_TP_BROADCAST_SHARD_BYTES=8388608
export VLLM_ASCEND_KV_TP_BROADCAST_ROOT_RANK=0   # 按你的拓扑调整，见下一节
```

## 针对你的拓扑（8 卡 / 2 个 dp / 只有 npu0、npu1 挂 RDMA 网卡）配置 ROOT_RANK

**关键事实：`tp_rank` 是"本 DP 副本自己的 TP 组内编号"，不是全局 rank。**
vLLM 的并行布局是 `DP x TP`（`vllm/distributed/parallel_state.py` 里
`initialize_model_parallel`），global rank 按 `dp_rank * tp_size + tp_rank`
连续分块：

```
dp_rank=0 的 TP 组 = global rank [0, 1, 2, 3]   (tp_rank 0,1,2,3)
dp_rank=1 的 TP 组 = global rank [4, 5, 6, 7]   (tp_rank 0,1,2,3)
```

如果 `global_rank == 物理 NPU 编号`（最常见的默认绑定方式），那么
**dp1 这个 TP 组里根本不存在 npu0/npu1**——它的 4 个 rank 对应的物理卡是
npu4~7，local `tp_rank==0` 落在 npu4 上，而不是你说的"亲和"的 npu1。
这种默认绑定下，`ROOT_RANK` 全部保持默认值 `0` **对 dp1 是不对的**：
dp1 的广播根会选中 npu4（远离 RDMA 网卡的卡），完全没解决你的问题。

所以要让「dp0 用 npu0 广播、dp1 用 npu1 广播」成立，有两条路，二选一：

1. **（推荐）改变物理设备绑定，让每个 DP 副本的 local tp_rank 0 本来就
   落在亲和的 NPU 上**，例如让 dp1 这组进程绑定的物理卡顺序是
   `[1, 3, 5, 7]`（而不是默认的 `[4,5,6,7]`），这样 tp_rank0 自然就是
   npu1。这通常通过你启动 worker 时设置的 `ASCEND_RT_VISIBLE_DEVICES`
   （或者 rank table / Ray placement group 里的设备顺序）来做，**不需要
   改这份补丁的代码**，`ROOT_RANK` 全部保持默认 `0` 即可。
2. **如果物理设备绑定没法按上面这样调整**（比如受限于其他约束，
   dp1 的 4 个 worker 物理上就是 npu4~7，npu1 根本不在这个 TP 组里），
   那么**这份补丁目前的设计是无法满足"跨 DP 组借用另一组的 NPU 做广播
   根"这个需求的**——HCCL/HCCS 广播只能在同一个 TP 通信组内部的 rank
   之间做，npu1 如果不属于 dp1 的 TP 组，就不可能成为 dp1 的广播根。
   这种情况下 dp1 只能退而求其次：在 dp1 自己的 4 张卡（npu4~7）里选一
   个相对离 RDMA 最近的（哪怕也不理想）作为 `ROOT_RANK`，设置方式是
   每个 worker 进程启动前按自己的 `tp_rank`/物理设备号设置：
   ```bash
   # 伪代码：在你的启动脚本里，按进程算出 ROOT_RANK 再 export
   # dp0 的 4 个 worker: ROOT_RANK 都设为 0 (local tp_rank0 = npu0，已经是亲和卡)
   # dp1 的 4 个 worker: 从 npu4~7 里选一个最近似亲和的 local tp_rank 设为 ROOT_RANK
   export VLLM_ASCEND_KV_TP_BROADCAST_ROOT_RANK=<该进程所在DP组里最优local tp_rank>
   ```
   这种情况下收益会小于"亲和卡能复用"的理想场景，因为广播根本身读远端
   时也要走非亲和的 RDMA 路径。

**怎么验证你当前到底是哪种情况**：在每个 worker 进程启动时打印一下
`tp_rank` 和它实际绑定到的物理 NPU 设备号（例如 `torch_npu.npu.current_device()`
或者你启动脚本里设置的可见设备顺序），确认 dp0/dp1 各自的 `tp_rank==0`
进程是不是真的分别落在 npu0/npu1 上。如果是，直接用默认配置（全部
`ROOT_RANK=0`）即可，不需要额外设置；如果不是，按方案 1 调整设备绑定
优先于方案 2 改 `ROOT_RANK`。

## 重要限制（上线前务必确认）

1. **只对"逐 rank 内容相同"的 KV 组安全。** 这在 MLA/DSA 类 latent cache
   （DeepSeek V2/V3/V4 等）、TP 内不切分该组的情况下成立；对普通
   GQA/MHA 按 head 切分的 KV 组是**不**成立的——如果误把这种组加进
   `VLLM_ASCEND_KV_TP_BROADCAST_GROUPS`，会把 rank0 的那一份 shard
   广播给所有 rank，**这是静默的数据错误，不会报错也不会崩溃**，
   只会导致该 KV 组对应的那部分计算结果不对。这一版本没有自动探测
   「这个 group 是不是真的处处相同」的信号，所以完全靠你显式配置，
   请对照你的 attention 实现（DeepSeek V4 Flash 的 MLA 部分）确认。
2. `dcp_size > 1` 时该优化会被自动强制关闭（即使设置了环境变量），
   因为 DCP 可能会把一个本应"TP 内相同"的组再按上下文维度切分，
   这时广播就不再安全。`pcp`（如果你的版本有这个并行维度）本补丁
   没有读取这个信号，如果你用了 PCP 请先手动确认再开启。
3. **这版本没有在真实昇腾硬件上跑过**——沙箱里没有 NPU/torch_npu，
   所有改动都是纯代码走查 + Python 语法编译检查（已经跑通，见下）。
   上线前务必按下面"验证步骤"先做正确性验证，再做性能验证。
4. 错误处理是"整批成败"粗粒度的，不是按 key 精细处理；如果广播状态
   显示失败，这批 broadcast 组的所有 block 都会被标记失效走重算路径，
   不会尝试区分到底是哪个 key 出的问题。

## 怎么用

1. 把这个目录（`tp_kv_broadcast.py` / `*.patch` / `apply_tp_broadcast_patch.sh`）
   拷贝到你的服务器上，放在任意位置，比如 `~/mooncake_tp_broadcast_patch/`。
2. 在服务器上执行：
   ```bash
   chmod +x ~/mooncake_tp_broadcast_patch/apply_tp_broadcast_patch.sh
   ~/mooncake_tp_broadcast_patch/apply_tp_broadcast_patch.sh /path/to/your/vllm-ascend
   ```
   脚本会：
   - 检查目标目录看起来像 vllm-ascend 仓库；
   - 如果能读到 git HEAD，校验是不是 `50e0ce60*`，不是的话给出警告但不会
     强行退出（你的 checkout 如果只有少量本地改动，补丁多半还能打上）；
   - 先用 `git apply --check` 两个 patch，任何一个打不上就直接中止，
     不会碰你的文件；
   - 真正应用两个 patch，拷贝新文件 `tp_kv_broadcast.py`；
   - 用 `python3 -m py_compile` 对三个文件做语法检查。
3. 设置上面两个环境变量，重启你的 vLLM 服务。
4. 默认状态下（不设置环境变量）补丁**完全不改变现有行为**——这是故意的，
   方便你先只打补丁、不开启功能，跑一轮回归确认没有引入问题，再单独
   决定什么时候打开环境变量。

## 建议的验证步骤

**正确性验证（必须先做）：**
- 找一个短请求，分别在"关闭广播"和"开启广播"两种配置下跑一遍，
  对比写入 KV cache 的内容是否逐字节一致（可以在 `m_store.get` 前后
  加临时的 debug dump，或者用你之前抓的 KV I/O trace 工具跑一遍离线
  重放对比）。
- 边界情况重点测：请求被截断/部分命中、某个 block 在远端缺失（制造
  `ret != 0` 的场景）、以及 `tp_mismatch` 分支（这条分支完全没有改动，
  但建议确认一下确实没被新代码路径影响到）。

**性能验证：**
- 复用你之前的 `tools/kv_io_bench` 工具重新跑一遍 "01 npu + 23 npu
  同时读"的那个带宽不打满的场景，对比开启前后 RDMA 网卡带宽利用率和
  端到端加载延迟。预期：RDMA 侧的并发读请求数从 4 降到 1（只有 rank0
  读远端），HCCS 侧多了一次广播流量，整体应该能让 01/23 混跑时的
  有效带宽接近单独跑 01 或 23 时的水平。

## 文件清单

```
mooncake_tp_broadcast_patch/
├── tp_kv_broadcast.py              # 新文件，直接拷贝到目标路径
├── kv_transfer.py.patch            # unified diff, git apply 使用
├── envs.py.patch                   # unified diff, git apply 使用
├── apply_tp_broadcast_patch.sh     # 一键应用脚本
└── README_tp_broadcast.md          # 本文件
```

补丁已经过本地验证：在重建的、内容与你固定 commit 一致的文件树上，
`git apply --check` 两个 patch 均成功，`git apply` 后文件内容与预期
逐行一致，且 `python3 -m py_compile` 对三个文件（`kv_transfer.py` /
`envs.py` / `tp_kv_broadcast.py`）编译通过，没有语法错误。**但没有
在真实 vllm-ascend 运行环境（含 torch_npu / Mooncake / 真实 NPU）里
跑过**，这是你在服务器上验证时需要重点关注的部分。

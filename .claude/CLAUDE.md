# SGLang — Flat Memory System 子项目规范

> 本文档是 SGLang 在 KVCache Flat Memory System 项目中的开发规范。
> 上层规范: `/home/huangxin/code_list/flat-memory-system/.claude/CLAUDE.md`
> 版本: SGLang v0.5.9 (commit 884568516)

---

## 0. 本文档范围

本文档仅规范 SGLang 在 Flat Memory System 项目中涉及的模块。SGLang 的通用开发规范（JIT Kernel、sgl-kernel 等）请参考 `.claude/skills/` 下的 SKILL.md 文件。

---

## 1. SGLang 在 Flat Memory 项目中的角色

SGLang 是 LLM 推理框架，在 Flat Memory 项目中承担：

| 职责 | 说明 |
|------|------|
| **PD 分离调度** | Prefill (GPU 0) / Decode (GPU 1) 分离，KVCache 在两个 GPU 之间通过 RDMA 传输 |
| **KVCache 生成** | 模型推理过程中产生 KVCache，由 Prefill 节点写入存储 |
| **HiCache 缓存管理** | 通过 HiRadixCache 管理 GPU HBM / Host DRAM / Storage (SSD) 三级缓存 |
| **Benchmark 测试** | 测量 TTFT、TBT、吞吐量、缓存命中率等指标 |

---

## 2. Flat Memory 相关的核心文件

### 2.1 KVCache 缓存管理

| 文件 | 说明 | Flat Memory 改动点 |
|------|------|-------------------|
| `python/sglang/srt/mem_cache/hiradix_cache.py` | HiRadixCache 核心 — GPU/Host/Storage 三级 RadixTree 缓存 | `evict_host()` (L891) 仍用 `children.pop()` (L916) 删除节点；**无需** Ghost Node 改造 — 取回不依赖 RadixTree 残留节点 |
| `python/sglang/srt/mem_cache/hiradix_cache.py:1264` | `prefetch_from_storage()` — 从 SSD/DRAM 预取 KVCache 到 Host DRAM | **已激活**（Issue #4 修复后）；通过 `token_ids` 重算 prefix-hash 链去 Mooncake 查询 |
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixTree 基类 — `match_prefix()` 前缀匹配逻辑 | 无需改动 — prefetch 触发不依赖 `match_prefix()` 命中残留节点（见 §3.1） |

### 2.2 MooncakeStore 存储后端

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore` — HiCacheStorage 后端，桥接 SGLang 与 Mooncake DistributedStore |
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:635` | `MooncakeStore.get()` — 调用 Mooncake `batch_get_into()` 从 DRAM/SSD 读取 KVCache |

### 2.3 PD 分离 / RDMA 直传

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/disaggregation/mooncake/conn.py` | `MooncakeKVManager/Sender/Receiver` — GPUDirect RDMA KVCache 直传 |
| `python/sglang/srt/distributed/mooncake_utils.py` | `MooncakeTransferEngine` — Transfer Engine Python 封装 |

### 2.4 调度器

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/managers/scheduler.py:1656` | `_prefetch_kvcache()` — 预取触发入口 |
| `python/sglang/srt/managers/scheduler.py:1671` | 触发条件 `req.last_node.backuped or req.last_node is root_node` — root 分支从 `token_ids` 重算 hash 链（PR #19663），即使节点被 `children.pop()` 删光仍能触发 |

---

## 3. 已知问题 (Flat Memory 相关)

### 3.1 Ghost Node 问题 (P0) — ✅ 已修复（通过 PR #19663 root_node 特判，非 Ghost Node 改造）

**原始根因:** `hiradix_cache.py` 的 `evict_host()` (L891) 使用 `children.pop(key)` (L916) 从 RadixTree 中彻底删除已驱逐节点，导致 `_prefetch_kvcache()` 的旧触发条件 `req.last_node.backuped` 永远为 False。

**原始影响:** SSD/DRAM 中的 KVCache 无法被取回，`cached_tokens_storage` 始终为 0。

**实际修复方案（已落地）:** **保留 `children.pop()` 不动**，改为在触发条件上加 root_node 特判（`scheduler.py:1671`，上游 PR #19663）：

```python
# scheduler.py:1671
if req.last_node.backuped or req.last_node is self.tree_cache.root_node:
    # root 分支：节点被 evict 删光后 last_node 退化为 root，
    # 此时从 req.fill_ids (token_ids) 重算 SHA256 prefix-hash 链去 Mooncake 查询
    last_hash = req.last_host_node.get_last_hash_value()
    ...
    self.tree_cache.prefetch_from_storage(req.rid, req.last_host_node,
                                          new_input_tokens, last_hash, ...)
```

**关键认知:** KVCache 取回**不依赖 RadixTree 是否残留节点**，而是靠 **prefix hash**：SGLang 用 token_ids 重算 hash → Mooncake Master 用该 hash 查询 → `is_local_disk_replica()` 决定从 **mooncake-DRAM**（RDMA 直读）还是 **mooncake-SSD**（RPC + preadv + RDMA 取回）读取。因此原先设想的「Ghost Node 标记 + 扩展 `match_prefix()`」方案**没有采用、也不需要**。

**关键代码位置:**
- `scheduler.py:1671` — 修复点（root_node 特判）
- `evict_host()`: L891-925 — 维持 `children.pop()` 原样
- `prefetch_from_storage()`: hiradix_cache.py:1264 — 修复后已激活

### 3.2 Duplicate DRAM 问题 (P0)

**根因:** `write_through` 无条件将 KVCache 写入 MooncakeStore DRAM。当 GPU HBM 满触发 `evict_to_host()` 后，同一份数据在 HiRadixCache L2 DRAM 和 MooncakeStore DRAM 中各存一份。

**影响:** 40GB DRAM 实际有效容量仅 20GB。

**修复方案:** Flat Memory Manager 统一管理 DRAM 池，消除双写。

### 3.3 索引层级

```
L1 (GPU HBM)   — RadixTree 索引（hiradix_cache.py）
L2 (Host DRAM)  — RadixTree 索引（与 L1 共用同一棵树）
L3 (SSD/DRAM)   — Prefix Hash 索引（Mooncake Master 维护）

L1/L2 与 L3 的衔接（已修复，§3.1）:
  evict_host() 的 children.pop() 仍会清空 L1/L2 RadixTree 节点，
  但取回不依赖残留节点 —— scheduler.py:1671 在 last_node==root 时
  从 token_ids 重算 prefix-hash 链，直接用 hash 查询 L3，打通 L1/L2 → L3。
```

---

## 4. Benchmark 配置

| 参数 | 值 |
|------|---|
| 模型 | `meta-llama/Llama-3.1-8B-Instruct` |
| Benchmark 工具 | `python -m sglang.bench_serving` |
| 数据集 | ShareGPT / GSP (Generated Shared Prefix) |
| Prefill 端口 | 30010 (bootstrap 8998) |
| Decode 端口 | 30011 |
| GPU 绑定 | Prefill: GPU 0 + mlx5_0, Decode: GPU 1 + mlx5_1 |

**关键环境变量:**

| 变量 | 说明 |
|------|------|
| `SGLANG_HICACHE_STORAGE` | 存储后端类型 (mooncake) |
| `SGLANG_HICACHE_WRITE_POLICY` | 写入策略 (write_through / write_back) |
| `MOONCAKE_CONFIG_PATH` | Mooncake 配置文件路径 |

---

## 5. 代码修改约束

- SGLang 代码修改在 `third_party/sglang/` 中进行
- 修改必须与顶层 CLAUDE.md 和 Mooncake CLAUDE.md 保持一致
- 所有 Flat Memory 相关修改需标注 `# FLAT_MEMORY:` 注释前缀，便于追踪
- 不得修改 SGLang 的通用推理逻辑（attention、MLP 等），仅修改缓存管理和存储层

---

## 6. 详细文档参考

| 文档 | 说明 |
|------|------|
| [顶层规范](../../../.claude/CLAUDE.md) | Flat Memory System 项目总规范 |
| [Issues 跟踪](../../../.claude/issues.md) | 待解决问题清单 |
| [Ghost Node 分析](../../../experiments/experiment_4_single_node_Flat_Memory/ghost_node_and_ssd_kvcache_reuse_analysis.md) | Ghost Node + Duplicate DRAM 根因分析 |
| [PD 分离文档](../../docs/advanced_features/pd_disaggregation.md) | SGLang PD 分离官方文档 |

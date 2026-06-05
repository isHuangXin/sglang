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
| `python/sglang/srt/mem_cache/hiradix_cache.py` | HiRadixCache 核心 — GPU/Host/Storage 三级 RadixTree 缓存 | **Ghost Node 修改点**: `evict_host()` (L839) 需保留节点索引而非 `children.pop()` |
| `python/sglang/srt/mem_cache/hiradix_cache.py:1163` | `prefetch_from_storage()` — 从 SSD 预取 KVCache 到 Host DRAM | 当前为死代码（Ghost Node 导致），修复后可激活 |
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixTree 基类 — `match_prefix()` 前缀匹配逻辑 | 需扩展以识别 Ghost Node 并触发 storage prefetch |

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
| `python/sglang/srt/managers/scheduler.py:1656` | `_prefetch_kvcache()` — 预取触发入口，检查 `req.last_node.backuped` |

---

## 3. 已知问题 (Flat Memory 相关)

### 3.1 Ghost Node 问题 (P0)

**根因:** `hiradix_cache.py` 的 `evict_host()` 使用 `children.pop(key)` 从 RadixTree 中彻底删除已驱逐节点。

**影响:** SSD 中的 KVCache 永远无法被 `match_prefix()` 发现，`cached_tokens_storage` 始终为 0。

**修复方案:** 将 `children.pop(key)` 替换为 Ghost Node 标记（保留节点、清空数据引用），使 `match_prefix()` 能匹配到 Ghost 节点并触发 `prefetch_from_storage()`。

**关键代码位置:**
- `evict_host()`: L839-871 — 修改为 Ghost Node 标记
- `match_prefix()`: 需扩展识别 Ghost Node
- `_prefetch_kvcache()`: L1656 — 已有预取逻辑，修复后自动激活

### 3.2 Duplicate DRAM 问题 (P0)

**根因:** `write_through` 无条件将 KVCache 写入 MooncakeStore DRAM。当 GPU HBM 满触发 `evict_to_host()` 后，同一份数据在 HiRadixCache L2 DRAM 和 MooncakeStore DRAM 中各存一份。

**影响:** 40GB DRAM 实际有效容量仅 20GB。

**修复方案:** Flat Memory Manager 统一管理 DRAM 池，消除双写。

### 3.3 索引层级

```
L1 (GPU HBM)   — RadixTree 索引（hiradix_cache.py）
L2 (Host DRAM)  — RadixTree 索引（与 L1 共用同一棵树）
L3 (SSD)        — Prefix Hash 索引（Mooncake Master 维护）

当前问题: evict_host() 的 children.pop() 导致 L1/L2 RadixTree 与 L3 Prefix Hash 断裂
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

# SGLang — Flat Memory System 子项目规范

## Author

Xin Huang — [@isHuangXin](https://github.com/isHuangXin)

## 范围与基线

本文件是 Flat Memory 项目对上游 SGLang 规范的补充。上层规范位于主仓库 `.claude/CLAUDE.md`；保留并遵守本目录的上游 `.claude/rules/` 和组件 skills。

- 当前迁移分支：`upstream-v0.5.19-flatcake-b200`。
- 上游起点：`upstream-v0.5.19`，`0bcd822377da7b5718e674eaf9c870d349424dd1`。
- Flat 来源：`flat-memory-system-sglang-v0.5.9`，tip `e4af9db2bb7cff3222e96301199f1d8b81c11995`。
- 项目增量基线：`bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`，其后的 27 个项目提交按最终功能迁移；不重放已被上游覆盖的 release 补丁。
- 保留分支 `upstream-v0.5.19-flatcake-b200-all-in-one` 的生产适配可复用，但其测试结果不能代替本分支的真实 GPU 验证。

## 修改约束

- SGLang 修改留在此子模块；原生 Flat 存储及绑定修改留在主仓库；Mooncake 修改留在对应子模块。
- Flat 专有修改使用 `# FLAT_MEMORY:` 标记。这是项目针对上游 comment tag 规则的专用补充；注释仍应简洁、英文、说明现存约束。
- 不改写通用 attention、MLP、模型计算和量化内核来掩盖缓存适配问题。
- 复用上游统一缓存、pool metadata 和生命周期接口，不整文件覆盖为 v0.5.9 实现。
- 不将兼容性、CPU 模拟测试或配置识别等同于真实模型/GPU 验证成功。

## 缓存与存储接入

v0.5.19 默认使用 `UnifiedRadixCache`，仅修改旧 `hiradix_cache.py` 不会接入默认服务路径。

| 位置（`python/sglang/` 下） | 职责 |
|---|---|
| `srt/mem_cache/registry.py`、`flat_memory_factory.py` | 注册并构造 Flat 路径 |
| `srt/mem_cache/flat_memory_cache.py` | scheduler 线程上的异步预取、完成共识、私有分配/发布和生命周期 |
| `srt/mem_cache/unified_cache/unified_cache_linker.py` | 统一树的锁、split、prepare/commit 和 offload 结构 |
| `srt/mem_cache/storage/flat_memory/` | host/v1/v2 adapter、GPU payload、direct linker 与介质 I/O |
| `srt/mem_cache/hybrid_cache/linker_pool_assembler.py` | 真实 MHA/DSA/V4 device pool 布局 |
| `srt/managers/cache_controller.py`、`srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | host 路径的有序 PrefetchAck 和 I/O ownership |
| `srt/managers/scheduler.py` | 预取/poll、admission、idle、flush、管理接口的接线 |

关键不变量：

1. GPU compat 路径不构造 HiCache host **payload** pool；Flat DRAM 与 cuFile 内部 bounce buffer 不属于该 host tier。
2. 私有 GPU 页面在读取和 TP 共识完成前不可发布到树。失败/取消先结束 I/O，再回收页面与 SWA 映射。
3. 备份 pending 与成功 committed 分开；拆分和驱逐不能把入队当成已持久化。
4. TP collective 在 scheduler 线程按确定次序执行，不能随 worker 完成次序调用。
5. root-prefetch 已存在于上游，保留新版完整 token/hash、extra key、cache salt 和 accessor 语义。
6. 不沿用旧版提前释放 host staging 的实现；host 释放以 DMA/storage completion 为界。
7. 普通 GPU cache flush 保留 Flat 内容；权重更新必须同时失效旧存储 KV。释放/重建物理池前必须先 detach Flat。

## 模型与配置边界

本次验收目标均为 B200 TP8：
- `/data/xinhuang/model_list/DeepSeek-V3.2`：TP8、64-token page。
- `/data/xinhuang/model_list/GLM-5.3`：TP8、64-token page。
- `/data/xinhuang/model_list/DeepSeek-V4-Flash-0731`：TP8、256-token page。

启动配置与只读预检入口位于主仓库 `experiments/experiment_4_single_node_Flat_Memory/flatcake_b200/`。运行时对三个模型均显式指定 `TP_SIZE=8`，覆盖 V4 profile 的 TP4 默认值。

- direct 支持边界为单机、PP1/DP1、TP1/4/8、Python tree、普通 page-aligned prefix reuse。
- 禁用未经适配的 CP/DCP、HiSparse、SSM、speculative/draft、请求快照和池地址重定位；不把普通 MHA 的 FP16/BF16 NHD 检查放宽后宣称支持 DSA/V4。
- DSA 必须保留实际 MLA/indexer/scale 数据；GLM 的物理 indexer 层由上游 pool 决定，不用固定的层数假设。
- V4 必须保留 SWA、C4/C128、indexer 及相应压缩状态。C128 state 在 256-token 对齐处的省略是条件不变量，不是任意断点恢复承诺。
- 每 rank 使用独立 manager/backing 路径，不能套用共享 Mooncake 的 rank0-only 写入策略。
- 当前原生 `gds_mode="compat"` 不支持 remote/persistence，不能称为严格硬件 GDS。

## 指标与工具

- Benchmark 实现在 `benchmark/serving.py` 与 `benchmark/datasets/`；`bench_serving.py` 保留上游兼容 shim。
- PD 槽位 0–6 保留上游含义；Flat 扩展使用版本与有效性字段，不能覆盖 image/audio/video。
- mixed 是 SSD 子集，不可作为第三层重复计数。缺失遥测用不可用值，不伪造零命中或零带宽。
- `/flat_memory/io_window` 使用全 rank 所有权校验、共同 drain、共享时钟边界；100ms bucket 聚合先跨 rank 合并再求峰值。
- 异步 profiler 仅把独立快照的导出放到有界任务；stop/flush 和通信同步不能随意移到后台线程。

## 验证与执行约定

本次只做必要语法/导入检查和三个模型依次 TP8 的 cold → offload/drain → GPU-only flush → storage restore 验证，不扩展测试矩阵。必须报告每个模型真实运行结果，不能用单个模型、配置测试或恒为零的 staging 指标代替完整验收。

使用当前容器 Python。依赖安装、原生构建、GPU 大模型运行和 SSD 目录使用遵守用户授权；不自动创建虚拟环境或替换系统 CUDA，不覆盖已有 KVCache。

不自动 commit、push 或将未提交实现误描述为已被子模块 gitlink 固定。

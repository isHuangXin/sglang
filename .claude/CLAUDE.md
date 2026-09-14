# SGLang — Native HiCache / Mooncake B200 基线

## Author

Xin Huang — [@isHuangXin](https://github.com/isHuangXin)

## 分支与迁移来源

- 当前分支：`upstream-v0.5.19-baseline-mooncake-b200`。
- 上游起点：`upstream-v0.5.19`，`0bcd822377da7b5718e674eaf9c870d349424dd1`。
- 复用已迁移的前 26 个共同项目提交，分叉点 `f40a5b2d00df95f74a7d3adacd6af1a34d81becf`。
- baseline 独有来源：`baseline-slang-v0.5.9-tcp-ack` 的 `16db2e906419756ba8eb6da0bae91cceb31c2435`，单独迁移为 `37881a4b27b2a5292291859f79c8a60ea795c9d7`。
- 必要的 v0.5.19 通用兼容独立提交，不合并、重写上述来源提交。
- 遵守主仓库 `.claude/CLAUDE.md` 和本目录上游 `.claude/rules/`、组件 skills。

## 基线运行架构

本分支用于原生 HiCache / Mooncake 分层存储基线，保留真实的 L2 Host 内存池：

`GPU HBM -> HiCache Host DRAM -> Mooncake DRAM / SSD`

- 默认缓存为 v0.5.19 的 `UnifiedRadixCache`；DSA/V4 等多池模型使用上游 Hybrid pool/controller 接口。
- Mooncake 通过 `HiCacheStorage` 后端接入；保留 v1/v2、多缓冲区与分组键语义。
- 不把 Flat 分支的“无 Host payload 层”约束套用到 Mooncake 基线，不自动启用 Flat direct radix factory。
- 共同历史保留了早期 Flat/GDS 代码，但本分支的 B200 验收路径是原生 HiCache + `mooncake`，不是 Flat 专用 GDS 实现。
- 不通过修改 attention、MLP 或量化内核掩盖缓存接口问题。

## Native I/O 统计

### L2 Host 拷贝

- 复用 `L2TransferEngine` 每次提交的独立完成事件，不复用 layer-completion ring 的事件计时。
- 使用 controller 的实际多池传输字节数；H2D/D2H 完成后各记一次，不能把入队、请求 token 数或 RPC 耗时当作已完成拷贝。
- `HostIOMetrics` 保留事件引用至完成，reset 前先 drain/synchronize，再递增 generation 并清零计数。
- Host 拷贝服务带宽为各 rank 已完成字节之和除以各 rank copy-stream 时间之和，不是 TP 聚合带宽。
- 平均 all-layer H2D rank-batch 时间不是单请求延迟，也不能直接加到 TTFT 上。

### Mooncake 存储

- 使用 `MooncakeDistributedStore.get_storage_io_stats()` 的累计完成态快照，要求已初始化并支持该 API 的 RealClient。
- `dram_read`、`dram_write`、`ssd_read`、`ssd_write` 各自提供 `bytes/ops/errors`；缺失或禁用时明确报告 unavailable。
- 不退回旧 submission 计数，不把 Python RPC 时间标成介质延迟，不用驱逐量估算 SSD 写带宽或使用容量。
- SSD 读统计原生 direct-I/O 正 CQE 字节，可能包含对齐放大；SSD 写统计 bucket 数据写入和 datasync 完成，发生在 metadata 持久化之前。
- `/storage_io/begin`、`/storage_io/end` 是 SSD owner 的 HTTP 窗口接口；EndWindow 本身不会 drain。
- SGLang 在结束窗口前完成 Host/备份的上层 drain；异步 Mooncake offload 仍按窗口内实际完成时间计数。
- 100 ms peak 是单个 storage-owner 进程的固定完成桶峰值，不能累加各 rank 独立峰值冒充全局峰值，也不能称作物理设备峰值。

## 生命周期与 TP 顺序

- 查询和读取可由本地 worker 并行执行，但结果及 `PrefetchAck` 按操作顺序发布，不能按完成先后进入 collective。
- D2H 入队到 storage owner 建立之间必须保持 ownership；只有对应 storage operation 完成或失败并产生 ACK 后才能释放 staging。
- Hybrid sidecar 写入仍按实际 rank/pool 所有权执行，不能因主 MLA KV 的 rank0 优化而跳过其他 rank 的 sidecar。
- 取消后的 worker、队列和 ACK 也要计入 pending；flush/停止不得在 native I/O 尚持有页面时释放内存。
- flush 首先确认无活跃请求，随后全 TP drain；超时返回失败，不清空仍被 I/O 持有的池。
- 保留上游 root-prefetch、完整 token/hash、cache salt、extra key 和 split/eviction 语义。
- PD 元数据槽位 0–6 保留上游含义，既有逐请求 device/host/storage 数组保留。
- profiler 只异步导出已停止的快照，正常退出时等待其导出任务结束。

## Benchmark 入口

`python -m sglang.bench_serving` 保留上游 shim，实现在 `python/sglang/benchmark/serving.py`。

| 参数 | 含义 |
|---|---|
| `--collect-hicache-io-metrics` | 原生 L2 Host 完成态统计 |
| `--collect-mooncake-io-metrics` | 原生 Mooncake DRAM/SSD 窗口与独立 L2 统计 |
| `--mooncake-master-host`、`--mooncake-metrics-port` | Master 容量指标 |
| `--mooncake-client-host`、`--mooncake-client-metrics-port` | SSD owner 原生窗口接口，默认端口 9301 |

原生采样要求 `--backend sglang`、server metrics、direct HiCache、PP1/DP1、无 PD；L2-only 使用 `layer_first`，Mooncake 使用 `page_first_direct`。

保留 `HICACHE_IO_DRAIN_TIMEOUT`（默认 120 秒）和 `HICACHE_FLUSH_TIMEOUT`（默认 150 秒）兼容控制，flush HTTP 预算必须大于 drain 预算。请求吞吐计时不包含事后的 drain/指标查询；指标缺失不能抹掉已完成请求的结果。

## B200 目标与验证边界

目标本地模型均为 TP8：
- `/data/xinhuang/model_list/DeepSeek-V3.2`：64-token page。
- `/data/xinhuang/model_list/GLM-5.3`：64-token page。
- `/data/xinhuang/model_list/DeepSeek-V4-Flash-0731`：256-token page。

代码迁移、语法检查或轻量导入检查不等同于这些模型已经跑通。真实 GPU 推理、Host/存储复用与原生绑定加载必须分别报告实际结果；不以旧日志或历史 benchmark 数据代替当前分支验收。

## 修改与数据约束

- SGLang 修改留在本子模块；不自动修改主仓库 gitlink、Mooncake 源码或原生库构建目录。
- 保留已有测试代码、补充测试数据、输入、结果和日志；不使用全目录 clean/reset 清理它们。
- 保留已发布的 Flat 分支及备份 stash，不回退或覆盖。
- 本次仅必要定向检查，不增加大测试集、性能矩阵或未授权的环境修改；commit/push/模型启动遵循用户明确授权。

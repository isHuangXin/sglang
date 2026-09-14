# SGLang — Tiered Mooncake GDS B200 子项目规范

## Author

Xin Huang — [@isHuangXin](https://github.com/isHuangXin)

## 范围与来源

本文件补充主仓库 `.claude/CLAUDE.md`；同时遵守本目录上游 `.claude/rules/` 与组件 skills。

- 当前分支：`upstream-v0.5.19-baseline-mooncake-gds-b200`。
- 上游基线：`upstream-v0.5.19`，`0bcd822377da7b5718e674eaf9c870d349424dd1`。
- 来源：`feature/mooncake-tiered-gds`，`fee32acf711c4e54b6d56bf64174e9ca91742fa6`。
- 来源的前 26 个项目提交复用已迁移的 `f40a5b2d0`；第 27 个 `e4af9db2b` 复用其独立迁移 `b2ba8750d`；第 28 个 `fee32acf7` 独立迁移为 `1639b2462`。
- 三模型 TP8、Unified 接线和最小启动配置另作兼容提交，不混入或替代来源提交。
- 普通 Mooncake baseline 独有的 `16db2e906` 不属于本来源序列；不整包引入其分支或 Flat 分支尾部实现。

## 分层与所有权

本分支是 **Tiered Mooncake GDS**，不是 Flat allocator：

- 后端为 `mooncake`，extra config 使用 `gds_mode="compat"`；保留真正的 HiCache Host tier。
- 保留 `init_hicache()`、`hicache_host_memory_mode="cache"`、原生 Host→Mooncake 写入与分层卸载。
- 不使用 Flat 工厂、编码地址或 `put_gpu_file` / `lookup_addresses` / `read_gpu`；不关闭 HiCache，不用 `buffer_only` 冒充分层模式。
- tiered runtime 与 Host controller 各负其责，不把它安装为会截断 Host 加载的 `cache.linker`。
- 保留 Mooncake v2 的 `_tag_keys()`、`_get_hybrid_page_component_keys()`、`_pack_multi_buffer_meta()` 及 Host multi-buffer 对象格式；GPU 读侧不能擅自改 key、顺序、长度或 namespace。
- GPU 直读完成后保留 L2 复用；Host 回填、备份与卸载保持单一所有权。

## GPU 读取与生命周期约束

- 原生 `batch_get_into_gpu` 每次最多 4096 个唯一 key，大小必须与存储对象的逻辑字节数一致，指针必须属于配置的 GPU。
- 使用有界连续 GPU 字节 staging，再按实际模型布局 scatter；不可丢失 FP8 scale、indexer 或压缩状态。
- 查询 worker 不分配 GPU 页面，不调用 TP collective；分配、共同恢复边界与发布由 scheduler 顺序协调。
- GPU 页面、锚点、staging 和完成事件在 native I/O、scatter 及全 rank 共识结束前保持私有且存活。
- 失败、取消、reset、detach 必须先完成本任务 I/O，再释放资源。退出时不得新增依赖已退出 rank 的 collective。
- 清理本次运行的 GPU/Host cache 不等于删除 Mooncake 对象；不得对共享或既有数据调用无范围的删除操作。
- 曾启用 tiered GDS 的实例拒绝原地权重更新或 pool 重建；即使已经 detach，也需使用新 namespace 重启，避免旧 native KV 或缓存的设备布局被重新使用。

## 三模型目标

| 本地目录（`/data/xinhuang/model_list/` 下） | TP | Page size | 必须保留的布局 |
|---|---:|---:|---|
| `DeepSeek-V3.2` | 8 | 64 | MLA、indexer 与 scale 原始字节 |
| `GLM-5.3` | 8 | 64 | 实际物理 indexer 层；跳过零行占位 |
| `DeepSeek-V4-Flash-0731` | 8 | 256 | SWA、C4/C128、indexer 和压缩状态 |

- KV 目标为 `fp8_e4m3`，按实际 pool 元数据处理原始字节，不依赖普通 MHA 的 FP16/BF16 假设。
- V4 仅在合法 256-token 对齐边界省略请求级 C128 state；共同恢复边界必须对所有 pool 和 TP rank 求交。
- 不改 attention、MLP 或量化计算内核来绕过缓存适配。
- 初始运行边界为单机、PP1/DP1、无 PD、无 speculative/CP/SSM；未经适配的组合明确拒绝。

## 指标与接口

- 原生 GPU 读取返回 `0=失败`、`1=DRAM`、`2=SSD`，不是读取字节数。
- 来源按对象和 TP rank 合并；mixed=3 只计入一次 SSD-dependent tokens，Host reload=4 更新此前来源；unknown/unavailable 不当作零或 SSD 命中。
- 只统计实际消费的前缀及新插入范围；保留逐请求 tier 数组与上游 PD 0–6 槽位。
- 同一请求的首次缓存前缀长度固定；后续 chunk/retraction 的真实 Host reload 只在该范围内重分类，不增加 `cached_tokens`，不把新 prefill 或生成 token 算作缓存命中。
- 分别报告已完成的 Host 复制、owner 原生存储 I/O 和 consumer cuFile 读取，不把提交态 transfer 字节或 RPC 耗时冒充介质 I/O。
- 使用专门的 GDS begin/end/window/clock API；不要与普通 `storage_io` 窗口混淆。窗口关闭不替代上层 drain。
- 只有已对齐的桶数据可聚合为跨 rank 峰值；不直接相加各 rank 独立峰值。
- Benchmark 实现在 `python/sglang/benchmark/`，保留 `bench_serving.py` shim；输出逻辑位于 `scheduler_components/output_streamer.py`，控制入口位于 `tokenizer_control_mixin.py`。

## 运行与验证边界

最小入口位于 `benchmark/mooncake_gds_b200/`：默认只预览，执行需显式指定同一套 native build/runtime 与新建的隔离数据、日志目录。拓扑为 master、本地 bucket owner 和 8 个 GPU consumers；使用 TCP RealClient、本地 GDS owner 许可及兼容模式，不复用 Flat-only 启动参数。

需要支持 `enable_gds`、`batch_get_into_gpu`、`get_gds_stats`、GDS begin/end/window/clock API 的真实绑定。现有构建目录名称或二进制字符串不能代替来源、ABI 和运行能力确认；缺少必要能力时应明确报错。

**当前状态：** 28 个来源提交已迁移，已加入三模型 TP8/Unified 兼容代码；三模型实机验收尚未执行。

2026-09-14 的最小检查在批准的 `/data/xinhuang/flat-kvcache-storage-dir` 下使用全新 4 KiB 文件：`cuFileDriverOpen=0`，`cuFileHandleRegister=5027`。日志指向 `udev property not found: ID_FS_USAGE md1`，当前容器缺少 `/run/udev`。记录保留在 `gds-handle-probe-20260914-pnr2f8ji/`。不得伪造 udev 数据、自动挂载或重建容器来绕过此问题；环境调整由用户确认并先保全仓库与未发布数据。

不新增大型测试套件、性能矩阵或长压测。只做必要语法/导入检查；环境修复后，逐模型做少量生成、真实 SSD 卸载与恢复、Host 复用和有界退出验证。仅服务启动或 HBM/DRAM 命中不证明 SSD GDS 路径通过；compat 也不等于严格硬件 GDS。

## 修改与交付

SGLang 修改留在本仓库，原生 Mooncake 与主仓库变更分开处理。项目专用注释按约定使用 `# FLAT_MEMORY:`，保持简短英文；保留上游命名、msgspec 和 RuntimeContext 约束。

所有补充测试代码、数据、结果、日志和已有分支均保留。不自动 push，不改主仓库 gitlink，不清理既有缓存。报告各模型的实际通过/阻塞状态，不把配置识别或静态检查当作实机成功。

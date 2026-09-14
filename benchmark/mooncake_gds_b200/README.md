# B200 Tiered Mooncake GDS: minimal launch entry

## Author

Xin Huang — [@isHuangXin](https://github.com/isHuangXin)

## Current status

**Hardware validation is blocked, not passed.** A fresh 4 KiB `O_DIRECT` probe on
`/data/xinhuang/flat-kvcache-storage-dir` (`/dev/md1`, ext4), with compatibility
mode forced, returned `cuFileDriverOpen=0` and `cuFileHandleRegister=5027` on
2026-09-14. The log reports missing `ID_FS_USAGE` for `md1`; the container has no
`/run/udev`. Evidence remains at:

`/data/xinhuang/flat-kvcache-storage-dir/gds-handle-probe-20260914-pnr2f8ji/cufile.log`

Do not start these models until the real device metadata is available and a
separate cuFile probe succeeds. Do not fabricate udev records or recreate a
container before preserving repositories and unpublished data. The probe source
is `build-flatcake-b200/cufile_handle_probe.cpp` in the parent repository.

The existing Mooncake `build-baseline-mooncake-gds-b200` directory is not a ready
runtime package: it lacks `mooncake_client`, and its recorded source versions
are inconsistent. Directory names and exported method names do not prove native
binary provenance or successful GDS I/O.

## Topology and profiles

Use one master, one local TCP bucket-storage owner, and one TP8 SGLang server
with eight RealClient consumers. Clients use loopback endpoints and share the
owner's local SSD root. No router, PD separation, Flat factory, or buffer-only
Host mode is used. Master RPC and metrics/admin explicitly bind to loopback;
keep all test services within the intended local environment.

| Model under `/data/xinhuang/model_list/` | TP | Page size | KV dtype |
| --- | ---: | ---: | --- |
| `DeepSeek-V3.2` | 8 | 64 | `fp8_e4m3` |
| `DeepSeek-V4-Flash-0731` | 8 | 256 | `fp8_e4m3` |
| `GLM-5.3` | 8 | 64 | `fp8_e4m3` |

Activation dtype stays `auto`. Host cache is retained at 8 decimal GB/rank by
default, using `direct`, `page_first_direct`, `write_through`, `wait_complete`,
and backend `mooncake` with `gds_mode=compat`. Consumers have
`global_segment_size=0`; the default owner has 256 MiB DRAM, a 128 MiB offload
buffer and 8 GiB SSD. GDS staging is 128 MiB with 16 MiB maximum I/O.

The owner sets `MOONCAKE_ALLOW_LOCAL_GDS=1` and
`MOONCAKE_OFFLOAD_USE_URING=true`; the current parser expects `true`/`false`, not
historical `MOONCAKE_USE_URING=1`. All roles force cuFile compatibility mode,
which may use Host staging and is not a claim of direct-DMA GDS.

The server uses PP1/DP1, concurrency one, 4096 context tokens, 8192 total KV
tokens, 2048-token prefill chunks and disabled CUDA graphs. The benchmark sends
one deterministic tokenized prefix (1024 + 64 tokens), requests eight output
tokens and performs no warmup. These are launch candidates, not measured
capacity or performance recommendations; a tiny request may not spill to SSD.

## Preview and later execution

Preview uses only the standard library and writes no files:

```bash
python3 -B benchmark/mooncake_gds_b200/launch.py --help
python3 -B benchmark/mooncake_gds_b200/launch.py server \
  --model DeepSeek-V3.2 --run-id preview-v32
python3 -B benchmark/mooncake_gds_b200/launch.py benchmark \
  --model GLM-5.3 --run-id preview-glm --sample ssd-repeat
```

Before execution, supply explicit `--native-build`, `--native-runtime` and
`--python` paths. The prepared runtime must contain:

- `bin/mooncake_master` and `bin/mooncake_client` from the intended native source;
- `python/mooncake/__init__.py` and exactly one `python/mooncake/store*.so`;
- required shared libraries under `python/mooncake/` or `lib/`.

The selected Python retains its normal user-site dependencies, including the
prepared Torch/Transformers stack. The launcher prepends the explicit native
and SGLang Python paths, and prepends native library directories while retaining
existing CUDA/library locations. It does not install or discover another runtime.

Execution checks actual backing-device udev metadata, native GDS/CUDA build
options, executables, and the selected binding's seven GDS APIs. These are
necessary checks only: native provenance and successful cuFile operation remain
separate prerequisites. The source-only commit originally supports MHA TP1/4;
the separate Unified/TP8 compatibility must be present before using these profiles.

After readiness is established, add `--execute` to **one role at a time**:
`master`, then `owner`, then `server`, in separate foreground terminals. Wait for
each service before proceeding. Use identical model, run ID, native/Python paths,
ports and budgets for every role in that run; benchmark samples use the same
arguments. The launcher never starts, stops, restarts or flushes other roles.

Each run has a new directory beneath `/data/xinhuang/flat-kvcache-storage-dir/`,
with immutable `run.json`, a new `ssd/`, and exclusive `roles/ROLE[-SAMPLE]/`
manifest/log/result files. Existing runs, samples and caches are never overwritten
or deleted. Check that ports and GPUs are available before execution.

## Minimal validation after environment repair

Run the three models sequentially, preserving their evidence:

1. **Cold populate:** generate from the fresh run's prefix and wait for completed
   Host copies and write-through backups.
2. **SSD reuse:** confirm that the actual prefix objects have reached SSD. Use
   bounded population/capacity settings suited to the objects, not a long pressure
   loop. Drain, then clear only the dedicated server's HBM/L2 through the supported
   cache operation, preserving native Mooncake objects. Repeat the same prefix.
3. Check successful generation, all-eight-rank restoration, nonzero **consumer**
   cuFile reads, valid window identity and SSD-dependent provenance. Owner writes,
   ordinary HBM/DRAM hits and cumulative storage counters alone do not prove this.
4. **Host reuse:** use a supported GPU-only eviction or bounded workload while
   retaining L2, then confirm Host-copy completion and Host provenance. A direct
   warm repeat may hit HBM instead. If the required path is not observed, report
   it as unverified rather than inventing an API or a successful result.

`cold`, `ssd-repeat` and `host-reload` are labels only; the benchmark does not
force those lifecycle transitions. Never call unscoped `remove_all`, clear OS
caches, delete existing storage, or stop processes belonging to someone else.

## Source references

The launcher records parent tiered scripts at `472d2192cdf0b24e79ed5a12bff78dd06cde7325`,
SGLang contract source `fee32acf711c4e54b6d56bf64174e9ca91742fa6`, and native GDS
source `d7a0d18157f87143c75dc1ae92a95da488339822`. The historical Llama70B/BF16/TP4
and Flat-only launch defaults are not reused as B200 tiered configurations.

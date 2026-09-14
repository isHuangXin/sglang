#!/usr/bin/env python3
"""Preview one B200 tiered Mooncake GDS role; execute only after explicit gates."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

SGLANG = Path(__file__).resolve().parents[2]
PARENT_REF = "472d2192cdf0b24e79ed5a12bff78dd06cde7325"
SGLANG_REF = "fee32acf711c4e54b6d56bf64174e9ca91742fa6"
NATIVE_REF = "d7a0d18157f87143c75dc1ae92a95da488339822"
STORAGE_BASE = Path("/data/xinhuang/flat-kvcache-storage-dir")
PROFILES = {"DeepSeek-V3.2": 64, "DeepSeek-V4-Flash-0731": 256, "GLM-5.3": 64}
CAPABILITIES = (
    "enable_gds", "batch_get_into_gpu", "get_gds_stats", "begin_gds_io_window",
    "end_gds_io_window", "get_gds_io_window", "gds_io_clock_ns",
)
OLD_TOPOLOGY = (
    "experiments/experiment_2_single_node_HicacheRDMA_SSD_IO_uring/"
    "sglang_with_L2_host_dram_L3_SSD_GDS/"
)
BLOCKER = (
    "BLOCKED/unverified: 2026-09-14 forced-compat O_DIRECT probe on /dev/md1 "
    "ext4 returned DriverOpen=0, HandleRegister=5027; ID_FS_USAGE missing and "
    "/run/udev absent. No service/model start until genuine readiness is restored."
)


def absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("use an absolute path without '..'")
    return path


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=BLOCKER)
    parser.add_argument("role", choices=("master", "owner", "server", "benchmark"))
    parser.add_argument("--model", required=True, choices=PROFILES)
    parser.add_argument("--run-id", required=True, help="new identifier; same for this run's roles")
    parser.add_argument("--execute", action="store_true", help="opt in to ONE role; default is stdout-only JSON preview")
    parser.add_argument("--native-build", type=absolute, help="explicit, provenance-checked CMake build directory")
    parser.add_argument("--native-runtime", type=absolute, help="explicit prepared runtime containing bin/ and python/mooncake/")
    parser.add_argument("--python", type=absolute, help="explicit prepared SGLang Python executable")
    parser.add_argument("--sample", choices=("cold", "ssd-repeat", "host-reload"), default="cold", help="benchmark label only; NEVER flushes or forces spill")
    parser.add_argument("--host-gb", type=positive, default=8, help="retained HiCache decimal GB per TP rank")
    parser.add_argument("--owner-dram-bytes", type=positive, default=256 << 20)
    parser.add_argument("--ssd-bytes", type=positive, default=8 << 30)
    parser.add_argument("--master-port", type=positive, default=50071)
    parser.add_argument("--master-metrics-port", type=positive, default=9014)
    parser.add_argument("--owner-port", type=positive, default=50072)
    parser.add_argument("--owner-http-port", type=positive, default=9311)
    parser.add_argument("--server-port", type=positive, default=30074)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("--run-id must be 1-64 letters, digits, underscores or hyphens")
    ports = [args.master_port, args.master_metrics_port, args.owner_port,
             args.owner_http_port, args.server_port]
    if max(ports) > 65535 or len(set(ports)) != len(ports):
        parser.error("ports must be distinct and between 1 and 65535")
    if args.role != "benchmark" and args.sample != "cold":
        parser.error("--sample is only for benchmark")
    return args


def make_manifest(args: argparse.Namespace) -> dict:
    runtime = args.native_runtime or Path("/REQUIRED/native-runtime")
    python = str(args.python or Path("/REQUIRED/python"))
    run = STORAGE_BASE / f"tiered-mooncake-gds-b200-{args.model}-{args.run_id}"
    label = args.role + (f"-{args.sample}" if args.role == "benchmark" else "")
    work = run / "roles" / label
    model_path = f"/data/xinhuang/model_list/{args.model}"
    extra = {
        "standalone_storage": False,
        "master_server_address": f"127.0.0.1:{args.master_port}",
        "master_metrics_port": args.master_metrics_port,
        "local_hostname": "127.0.0.1", "metadata_server": "P2PHANDSHAKE",
        "protocol": "tcp", "device_name": "", "global_segment_size": 0,
        "check_server": False, "extra_backend_tag": run.name,
        "gds_mode": "compat", "gds_ssd_root": str(run / "ssd"),
        "gds_max_io_bytes": 16 << 20, "gds_stage_bytes": 128 << 20,
    }
    env = {
        "CUFILE_FORCE_COMPAT_MODE": "true",
        "PYTHONPATH": f"{runtime}/python:{SGLANG}/python",
        "LD_LIBRARY_PATH": os.pathsep.join(
            value for value in (f"{runtime}/python/mooncake", f"{runtime}/lib",
                                os.environ.get("LD_LIBRARY_PATH", "")) if value
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(work / "cache"), "HF_HOME": str(work / "cache/hf"),
        "TORCH_EXTENSIONS_DIR": str(work / "cache/torch_extensions"),
        "TRITON_CACHE_DIR": str(work / "cache/triton"), "TMPDIR": str(work / "tmp"),
    }
    if args.role == "master":
        command = [str(runtime / "bin/mooncake_master"),
                   "--rpc_address=127.0.0.1", "--metrics_host=127.0.0.1",
                   "--port", str(args.master_port),
                   "--metrics_port", str(args.master_metrics_port),
                   "--enable_metric_reporting=true", "--enable_disk_eviction=true",
                   "--enable_offload=true", "--global_file_segment_size", str(args.ssd_bytes),
                   "--eviction_high_watermark_ratio", "0.95"]
    elif args.role == "owner":
        env.update({
            "MOONCAKE_ALLOW_LOCAL_GDS": "1", "MOONCAKE_OFFLOAD_USE_URING": "true",
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
            "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "bucket_storage_backend",
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": extra["gds_ssd_root"],
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(args.ssd_bytes),
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(128 << 20),
        })
        command = [str(runtime / "bin/mooncake_client"), "--host", "127.0.0.1",
                   "--port", str(args.owner_port), "--master_server_address",
                   extra["master_server_address"], "--metadata_server", "P2PHANDSHAKE",
                   "--protocol", "tcp", "--device_names", "", "--global_segment_size",
                   str(args.owner_dram_bytes), "--enable_offload=true",
                   "--start_offload_rpc_server=true", "--enable_http_server=true",
                   "--http_port", str(args.owner_http_port)]
    elif args.role == "server":
        env.update({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                    "NCCL_P2P_DISABLE": "0", "NCCL_IB_DISABLE": "1"})
        command = [python, "-m", "sglang.launch_server", "--model-path", model_path,
                   "--trust-remote-code", "--dtype", "auto", "--kv-cache-dtype", "fp8_e4m3",
                   "--tp-size", "8", "--pp-size", "1", "--dp-size", "1",
                   "--host", "127.0.0.1", "--port", str(args.server_port),
                   "--mem-fraction-static", "0.80", "--max-total-tokens", "8192",
                   "--context-length", "4096", "--chunked-prefill-size", "2048",
                   "--max-running-requests", "1", "--disable-cuda-graph",
                   "--enable-metrics", "--enable-cache-report", "--enable-hierarchical-cache",
                   "--hicache-size", str(args.host_gb), "--hicache-host-memory-mode", "cache",
                   "--hicache-write-policy", "write_through",
                   "--hicache-io-backend", "direct", "--hicache-mem-layout", "page_first_direct",
                   "--page-size", str(PROFILES[args.model]), "--hicache-storage-backend", "mooncake",
                   "--hicache-storage-prefetch-policy", "wait_complete",
                   "--hicache-storage-backend-extra-config", json.dumps(extra, sort_keys=True)]
    else:
        command = [python, "-m", "sglang.benchmark.serving", "--backend", "sglang",
                   "--model", model_path, "--host", "127.0.0.1", "--port", str(args.server_port),
                   "--dataset-name", "generated-shared-prefix", "--tokenize-prompt",
                   "--gsp-system-prompt-len", "1024", "--gsp-question-len", "64",
                   "--gsp-output-len", "8", "--gsp-num-groups", "1",
                   "--gsp-prompts-per-group", "1", "--gsp-num-turns", "1", "--gsp-range-ratio", "1",
                   "--num-prompts", "1", "--request-rate", "1", "--max-concurrency", "1",
                   "--seed", "1", "--warmup-requests", "0",
                   "--collect-hicache-io-metrics", "--collect-mooncake-io-metrics",
                   "--collect-mooncake-gds-cache-metrics", "--collect-mooncake-gds-io",
                   "--mooncake-master-host", "127.0.0.1", "--mooncake-metrics-port", str(args.master_metrics_port),
                   "--mooncake-client-host", "127.0.0.1", "--mooncake-client-metrics-port", str(args.owner_http_port),
                   "--prefill-metrics-host", "", "--output-details", "--output-file", str(work / "result.jsonl")]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"role", "sample", "execute"}}
    return {
        "status": BLOCKER, "mode": "execute-requested" if args.execute else "preview-only",
        "role": args.role, "sample": args.sample if args.role == "benchmark" else None,
        "launcher_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "topology": "1 master + 1 local bucket owner + 8 TP RealClient GPU consumers; PP1 DP1",
        "config": config, "run_dir": str(run), "role_dir": str(work),
        "model_path": model_path, "model_config_path": model_path + "/config.json",
        "page_size": PROFILES[args.model], "backend_extra_config": extra,
        "environment_overrides": env, "command": command,
        "shell_command_reference_only": shlex.join(command),
        "log": str(work / "process.log"),
        "native_required_capabilities": CAPABILITIES,
        "sources": {
            "parent": {"revision": PARENT_REF, "paths": [OLD_TOPOLOGY + name for name in
                       ("config.sh", "run_master.sh", "run_mooncake_client.sh", "run_server.sh", "benchmark.sh")]},
            "sglang": {"revision": SGLANG_REF, "paths": ["python/sglang/srt/server_args.py",
                       "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py",
                       "python/sglang/bench_serving.py"], "benchmark_relocated_to": "python/sglang/benchmark/serving.py"},
            "native": {"required_revision": NATIVE_REF, "paths": ["mooncake-store/src/real_client_main.cpp",
                       "mooncake-store/src/config/file_storage_config.cpp", "mooncake-store/src/CMakeLists.txt",
                       "mooncake-integration/store/store_py.cpp"]},
        },
        "warnings": ["Profiles and hardware remain unverified; source fee alone lacks B200 core compatibility.",
                     "Tiny cold request does not guarantee SSD spill. Samples are labels, not lifecycle actions.",
                     "Native provenance and successful cuFile HandleRegister must be checked separately; udev/API presence is not proof of readiness.",
                     "No cache flush, deletion, service stop, build, install or mount is performed by this launcher."],
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def preflight(args: argparse.Namespace, manifest: dict) -> dict:
    """Read evidence first; never open a store, compile, or run a cuFile probe."""
    device = STORAGE_BASE.stat().st_dev
    udev = Path(f"/run/udev/data/b{os.major(device)}:{os.minor(device)}")
    require(udev.is_file(), f"GDS blocked: real backing-device udev record absent: {udev}; no service started")
    require("E:ID_FS_USAGE=filesystem" in udev.read_text().splitlines(),
            "GDS blocked: real udev record lacks ID_FS_USAGE=filesystem")
    require(all((args.native_build, args.native_runtime, args.python)),
            "require explicit --native-build, --native-runtime and --python; no runtime auto-discovery")
    cache = args.native_build / "CMakeCache.txt"
    cmake = cache.read_text().splitlines()
    require("STORE_USE_GDS:BOOL=ON" in cmake and "USE_CUDA:BOOL=ON" in cmake,
            "native build requires STORE_USE_GDS=ON and USE_CUDA=ON; no automatic build")
    native_files = [args.native_runtime / "bin" / name for name in ("mooncake_master", "mooncake_client")]
    require(all(path.is_file() and os.access(path, os.X_OK) for path in native_files),
            "prepared runtime needs executable bin/mooncake_master AND bin/mooncake_client; build owner target separately")
    require(args.python.is_file() and os.access(args.python, os.X_OK), "missing explicit Python executable")
    package = args.native_runtime / "python/mooncake"
    require((package / "__init__.py").is_file(), "runtime missing python/mooncake/__init__.py; not a prepared package")
    stores = list(package.glob("store*.so"))
    require(len(stores) == 1, "runtime must contain exactly one python/mooncake/store*.so")
    # Import only: no MooncakeDistributedStore instance/setup or CUDA operation.
    code = (
        "import json, pathlib, mooncake.store as s; "
        f"expected=pathlib.Path({str(stores[0])!r}).resolve(); "
        "assert pathlib.Path(s.__file__).resolve()==expected, 'unexpected Mooncake store import'; "
        f"missing=[n for n in {CAPABILITIES!r} if not hasattr(s.MooncakeDistributedStore,n)]; "
        "assert not missing, 'missing native GDS capabilities: '+','.join(missing); "
        "print(json.dumps({'store':str(expected)}))"
    )
    env = role_environment(manifest)
    result = subprocess.run([str(args.python), "-B", "-c", code], env=env, cwd="/",
                            text=True, capture_output=True, timeout=30)
    require(result.returncode == 0, "native import/capability check failed: " + result.stdout + result.stderr)
    return {"udev_record": str(udev), "native_import": result.stdout.strip(),
            "native_build": str(args.native_build.resolve()),
            "native_binaries": [str(path.resolve()) for path in native_files],
            "note": "Necessary checks only; source provenance/cuFile probe remain separately approved prerequisites."}


def role_environment(manifest: dict) -> dict:
    # Keep the selected Python/CUDA environment, not inherited serving configuration.
    env = {key: value for key, value in os.environ.items() if key in
           {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
            "CUDA_HOME", "CUDA_PATH"}}
    env.update(manifest["environment_overrides"])
    return env


def execute(args: argparse.Namespace, manifest: dict) -> None:
    run, work = Path(manifest["run_dir"]), Path(manifest["role_dir"])
    require(not work.exists() and not work.is_symlink(), "refusing existing role/sample path; choose a fresh run")
    if args.role == "master":
        require(not run.exists() and not run.is_symlink(), "refusing existing run/storage; choose a fresh --run-id")
    else:
        require((run / "run.json").is_file(), "start the explicit master role for this fresh run first")
        require(json.loads((run / "run.json").read_text()) == manifest["config"],
                "run configuration differs; never mix models, runtimes or port/budget settings")
    manifest["readiness_checks"] = preflight(args, manifest)
    if args.role in ("server", "benchmark"):
        require(Path(manifest["model_config_path"]).is_file(), "local model config.json missing")
        require((run / "ssd").is_dir(), "local owner has not created this run's common SSD root")
    if args.role == "master":
        run.mkdir()  # Exclusive: never reuse/delete any previous run.
        with (run / "run.json").open("x") as stream:
            json.dump(manifest["config"], stream, indent=2)
        (run / "roles").mkdir()
    work.mkdir()  # Every role/sample may be executed only once in this run.
    (work / "tmp").mkdir()
    if args.role == "owner":
        (run / "ssd").mkdir()
    with (work / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Executing only {args.role}; log: {manifest['log']}", flush=True)
    with Path(manifest["log"]).open("x") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.chdir(work)
    os.execve(manifest["command"][0], manifest["command"], role_environment(manifest))


def main() -> int:
    args = arguments()
    manifest = make_manifest(args)
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return 0
    try:
        execute(args, manifest)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

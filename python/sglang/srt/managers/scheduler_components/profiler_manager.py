from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
)

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import set_detailed_annotations_enabled
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_device
from sglang.srt.utils import is_mps, is_npu
from sglang.srt.utils.profile_merger import ProfileMerger
from sglang.srt.utils.profile_utils import ProfileManager
from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

_is_npu = is_npu()
_is_mps = is_mps()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    apply_torch_npu_patches(torch_npu, patches)
elif _is_mps:
    from sglang.srt.hardware_backend.mlx.profiler import apply_metal_profiler_patches

    apply_metal_profiler_patches()

logger = logging.getLogger(__name__)


def _export_profile_snapshot(
    *,
    profiler,
    trace_path,
    rpd_source,
    marker_path,
    required_markers,
    output_dir,
    profile_id,
    merge,
):
    # FLAT_MEMORY: Workers never touch serving collectives or live profiler state.
    marker = Path(marker_path)
    try:
        if profiler is not None:
            profiler.export_chrome_trace(str(trace_path))
        if rpd_source is not None:
            from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

            rpd_to_chrome_trace(str(rpd_source), str(trace_path))
        marker.write_text("ok")
        if merge:
            deadline = time.monotonic() + 300
            while True:
                states = [
                    Path(path).read_text() if Path(path).exists() else None
                    for path in required_markers
                ]
                if any(state and state != "ok" for state in states):
                    raise RuntimeError("A peer profiler export failed")
                if all(state == "ok" for state in states):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for peer profile exports")
                time.sleep(0.05)
            merged = ProfileMerger(output_dir, profile_id).merge_chrome_traces()
            logger.info("Profile merge completed: %s", merged)
        logger.info("Profile export completed: %s", trace_path)
    except Exception as exc:
        marker.write_text(f"error: {exc}")
        logger.exception("Asynchronous profile export failed")
        raise
    finally:
        if rpd_source is not None:
            Path(rpd_source).unlink(missing_ok=True)


@dataclass(kw_only=True)
class SchedulerProfilerManager:
    ps: Any
    dp_tp_cpu_group: Any
    get_forward_ct: Callable[[], int]

    def __post_init__(self) -> None:
        # FLAT_MEMORY: Export jobs retain stopped profilers, not mutable manager state.
        self._export_executor: ThreadPoolExecutor | None = None
        self._export_jobs: list[Future] = []
        self._export_errors: list[str] = []
        self.rpd_profile_path: str | None = None
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager = ProfileManager(
                ps=self.ps,
                cpu_group=self.dp_tp_cpu_group,
            )
            return

        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[Path] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profile_id: Optional[str] = None

        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None

        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None

        self.profile_by_stage: bool = False
        self.profile_in_progress: bool = False
        self.merge_profiles = False
        self.detailed_annotations: bool = False

        # For ROCM
        self.rpd_profiler = None

    def _init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool = False,
        profile_prefix: str = "",
        detailed_annotations: bool = False,
        profile_stages: Optional[List[str]] = None,
    ) -> ProfileReqOutput:
        if envs.SGLANG_PROFILE_V2.get():
            self.detailed_annotations = detailed_annotations
            return self._profile_manager.configure(
                output_dir=output_dir,
                start_step=start_step,
                num_steps=num_steps,
                activities=activities,
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=profile_by_stage,
                profile_id=profile_id,
                merge_profiles=merge_profiles,
                profile_prefix=profile_prefix,
                profile_stages=profile_stages,
                detailed_annotations=detailed_annotations,
            )

        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage
        self.merge_profiles = merge_profiles

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = Path(output_dir).expanduser()
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id
        self.profile_prefix = profile_prefix
        self.detailed_annotations = detailed_annotations

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.get_forward_ct() + 1)

        if num_steps:
            if self.profile_by_stage:
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
            elif start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                self.profiler_target_forward_ct = self.get_forward_ct() + num_steps + 1
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def _apply_detailed_annotations(self, enabled: bool) -> None:
        # Toggle the process-wide flag read by build_step_span_name; folds the
        # per-phase sq/sqsq/sqsk/sk aggregates (context c_ / generation g_)
        # into the step span while a detailed-annotation profile is active.
        set_detailed_annotations_enabled(enabled)

    def _start_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            self._apply_detailed_annotations(self.detailed_annotations)
            return self._profile_manager.manual_start()

        self._collect_export_jobs()
        backlog = torch.tensor([len(self._export_jobs)], dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                backlog, op=torch.distributed.ReduceOp.MAX, group=self.dp_tp_cpu_group
            )
        if int(backlog.item()) >= 2:
            return ProfileReqOutput(
                success=False,
                message="Two profile exports are still pending; retry after they complete.",
            )
        stage_str = f" for {stage.name}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }

        if current_platform.is_out_of_tree():
            if hasattr(
                torch.profiler.ProfilerActivity,
                current_platform.get_torch_profiler_activity_str(),
            ):
                activity_map[current_platform.get_torch_profiler_activity_str()] = (
                    current_platform.get_torch_profiler_activity()
                )
        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            activity_map["XPU"] = torch.profiler.ProfilerActivity.XPU
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:  # for ROCM
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",
            )

            if self.ps.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.dp_tp_cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
            self.profile_in_progress = True
        elif torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                on_trace_ready=(
                    None
                    if not _is_npu
                    else torch_npu.profiler.tensorboard_trace_handler(
                        str(self.torch_profiler_output_dir)
                    )
                ),
                experimental_config=(
                    None
                    if not _is_npu
                    else torch_npu.profiler._ExperimentalConfig(
                        export_type=torch_npu.profiler.ExportType.Text,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None,
                    )
                ),
            )
            try:
                self.torch_profiler.start()
            except RuntimeError as e:
                self.torch_profiler = None
                return ProfileReqOutput(success=False, message=str(e))
            self.profile_in_progress = True

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(
                max_entries=envs.SGLANG_MEM_PROFILE_MAX_ENTRIES.get()
            )
            self.profile_in_progress = True

        if "CUDA_PROFILER" in activities:
            if self.ps.gpu_id == get_device().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        self._apply_detailed_annotations(self.detailed_annotations)
        return ProfileReqOutput(success=True, message="Succeeded")

    def _merge_profile_traces(self) -> str:
        if not self.merge_profiles:
            return ""

        if self.ps.tp_rank != 0:
            return ""
        if self.ps.dp_size > 1 and self.ps.dp_rank != 0:
            return ""
        if self.ps.pp_size > 1 and self.ps.pp_rank != 0:
            return ""
        if self.ps.moe_ep_size > 1 and self.ps.moe_ep_rank != 0:
            return ""

        try:
            logger.info("Starting profile merge...")
            merger = ProfileMerger(self.torch_profiler_output_dir, self.profile_id)
            merged_path = merger.merge_chrome_traces()

            summary = merger.get_merge_summary()
            merge_message = (
                f" Merged trace: {merged_path} "
                f"(Events: {summary.get('total_events', '?')}, "
                f"Files: {summary.get('total_files', '?')})"
            )

            logger.info(f"Profile merge completed: {merged_path}")
        except Exception as e:
            logger.error(f"Failed to merge profiles: {e}", exc_info=True)
            return f" Merge failed: {e!s}"
        else:
            return merge_message

    def _collect_export_jobs(self, *, wait=False):
        pending = []
        for job in self._export_jobs:
            if not wait and not job.done():
                pending.append(job)
                continue
            try:
                job.result()
            except Exception as exc:
                self._export_errors.append(str(exc))
                logger.error("Profile export failed: %s", exc)
        self._export_jobs = pending

    def close(self):
        """Stop local recording and drain exports without shutdown collectives."""
        try:
            if envs.SGLANG_PROFILE_V2.get():
                if self._profile_manager.profiler is not None:
                    self._stop_v2_profile_locally(self._profile_manager.profiler)
                    self._profile_manager.profiler = None
                    self._apply_detailed_annotations(False)
            elif self.profile_in_progress:
                # FLAT_MEMORY: Shutdown may reach only a subset of the profiling ranks.
                self._stop_profile(synchronize=False)
        except Exception as exc:
            self._export_errors.append(str(exc))
            logger.exception("Failed to finalize local profiler during shutdown")
        finally:
            self._collect_export_jobs(wait=True)
            if self._export_executor is not None:
                self._export_executor.shutdown(wait=True)
                self._export_executor = None

    def _stop_v2_profile_locally(self, profiler):
        from sglang.srt.utils.profile_utils import (
            _ProfilerCudart,
            _ProfilerList,
            _ProfilerMemory,
            _ProfilerRPD,
            _ProfilerTorch,
        )

        if isinstance(profiler, _ProfilerList):
            for inner in profiler.inners:
                self._stop_v2_profile_locally(inner)
        elif isinstance(profiler, _ProfilerTorch):
            profiler.torch_profiler.stop()
            if not _is_npu:
                path = self._rank_trace_path(
                    output_dir=profiler.output_dir,
                    profile_id=profiler.profile_id,
                    prefix=profiler.output_prefix,
                    suffix=profiler.output_suffix,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                profiler.torch_profiler.export_chrome_trace(str(path))
        elif isinstance(profiler, _ProfilerRPD):
            profiler.rpd_profiler.rangePop()
            profiler.rpd_profiler.stop()
            profiler.rpd_profiler.flush()
            logger.warning(
                "RPD shutdown retained trace.rpd without cross-rank conversion"
            )
        elif isinstance(profiler, (_ProfilerMemory, _ProfilerCudart)):
            profiler.stop()
        else:
            raise TypeError(
                f"Unsupported profiler shutdown type: {type(profiler).__name__}"
            )

    def _trace_path(self, stage):
        return self._rank_trace_path(
            output_dir=self.torch_profiler_output_dir,
            profile_id=self.profile_id,
            prefix=self.profile_prefix,
            suffix=f"-{stage.name}" if stage else "",
        )

    def _rank_trace_path(self, *, output_dir, profile_id, prefix, suffix):
        parts = [profile_id, f"TP-{self.ps.tp_rank}"]
        for label, size, rank in (
            ("DP", self.ps.dp_size, self.ps.dp_rank),
            ("PP", self.ps.pp_size, self.ps.pp_rank),
            ("EP", self.ps.moe_ep_size, self.ps.moe_ep_rank),
        ):
            if size > 1:
                parts.append(f"{label}-{rank}")
        prefix = f"{prefix}-" if prefix else ""
        return Path(output_dir) / (prefix + "-".join(parts) + suffix + ".trace.json.gz")

    def _queue_export(
        self, *, profiler, trace_path, rpd_source, marker, markers, allow_merge=True
    ):
        self._collect_export_jobs()
        if self._export_executor is None:
            self._export_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="profile-export"
            )
        merge = (
            allow_merge
            and self.merge_profiles
            and all(
                (
                    self.ps.tp_rank == 0,
                    self.ps.dp_rank == 0,
                    self.ps.pp_rank == 0,
                    self.ps.moe_ep_rank == 0,
                )
            )
        )
        profile_key = f"{self.profile_prefix}-" if self.profile_prefix else ""
        profile_key += self.profile_id
        self._export_jobs.append(
            self._export_executor.submit(
                _export_profile_snapshot,
                profiler=profiler,
                trace_path=trace_path,
                rpd_source=rpd_source,
                marker_path=marker,
                required_markers=tuple(markers),
                output_dir=self.torch_profiler_output_dir,
                profile_id=profile_key,
                merge=merge,
            )
        )

    def _stop_profile(
        self, stage: Optional[ForwardMode] = None, *, synchronize: bool = True
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            self._apply_detailed_annotations(False)
            return self._profile_manager.manual_stop()
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self._trace_path(stage)
        profiler, rpd_source = self.torch_profiler, None
        # FLAT_MEMORY: Stop/flush uses the originating thread; only immutable exports run asynchronously.
        if profiler is not None:
            profiler.stop()
            if _is_npu:
                if synchronize:
                    torch.distributed.barrier(self.dp_tp_cpu_group)
                    self._merge_profile_traces()
                profiler = None
        if self.rpd_profiler is not None:
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()
            if synchronize:
                torch.distributed.barrier(self.dp_tp_cpu_group)
                if self.ps.tp_rank == 0:
                    rpd_source = (
                        self.torch_profiler_output_dir
                        / f".rpd-{uuid.uuid4().hex}.snapshot"
                    )
                    shutil.copyfile("trace.rpd", rpd_source)
            else:
                logger.warning(
                    "RPD shutdown retained trace.rpd without cross-rank conversion"
                )

        activities = self.profiler_activities or ()
        if "MEM" in activities:
            torch.cuda.memory._dump_snapshot(str(trace_path) + ".memory.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        if "CUDA_PROFILER" in activities and self.ps.gpu_id == get_device().base_gpu_id:
            torch.cuda.cudart().cudaProfilerStop()

        marker = str(trace_path) + f".{uuid.uuid4().hex}.export"
        has_export = profiler is not None or rpd_source is not None
        markers = [marker] if has_export else []
        if synchronize and self.merge_profiles:
            # All communication happens at the existing scheduler-side stop boundary.
            gathered = [None] * torch.distributed.get_world_size(self.dp_tp_cpu_group)
            torch.distributed.all_gather_object(
                gathered, marker if has_export else None, self.dp_tp_cpu_group
            )
            markers = [value for value in gathered if value is not None]
        self.torch_profiler = None
        self.rpd_profiler = None
        self.rpd_profile_path = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None
        self._apply_detailed_annotations(False)
        if has_export:
            self._queue_export(
                profiler=profiler,
                trace_path=trace_path,
                rpd_source=rpd_source,
                marker=marker,
                markers=markers,
                allow_merge=synchronize,
            )
        message = (
            "Profiler stopped; trace export queued asynchronously."
            if has_export
            else "Profiler stopped."
        )
        logger.info("%s Output: %s", message, trace_path)
        return ProfileReqOutput(success=True, message=message)

    def _profile_batch_predicate(self, batch: ScheduleBatch):
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager.step(forward_mode=batch.forward_mode)
            return

        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                if self.profiler_prefill_ct == 0:
                    self._start_profile(batch.forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.EXTEND)
            elif batch.forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        # force trace flush
                        self._stop_profile(stage=ForwardMode.EXTEND)
                    self._start_profile(batch.forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.DECODE)
            elif batch.forward_mode.is_idle():
                pass
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")
        else:
            # Check profiler
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.get_forward_ct()
            ):
                self._stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.get_forward_ct()
            ):
                self._start_profile()

    def _profile(self, recv_req: ProfileReq):
        if recv_req.req_type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage or recv_req.start_step:
                return self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.detailed_annotations,
                    recv_req.profile_stages,
                )
            else:
                configured = self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.detailed_annotations,
                )
                if not configured.success:
                    return configured
                return self._start_profile()
        else:
            return self._stop_profile()

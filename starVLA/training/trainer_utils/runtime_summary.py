import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _max(values):
    return max(values) if values else 0.0


@contextmanager
def timed_section(timings: dict, name: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - start_time


class RuntimeSummary:
    """Collect lightweight runtime statistics for a training run.

    This intentionally avoids profiler dependencies and records only values
    already available in the training loop.
    """

    def __init__(
        self,
        output_dir,
        per_loop_batch_size: int,
        total_batch_size: int,
        world_size: int,
        gradient_accumulation_steps: int,
        output_basename: str = "runtime_summary",
    ):
        self.output_dir = Path(output_dir)
        self.per_loop_batch_size = int(per_loop_batch_size)
        self.total_batch_size = int(total_batch_size)
        self.world_size = int(world_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.output_basename = output_basename
        self.start_time = time.perf_counter()
        self.train_loop_start_time = None
        self.train_loop_end_time = None
        self.setup_times = {}
        self.data_times = []
        self.model_times = []
        self.step_times = []

    def record_setup_time(self, name: str, duration: float):
        self.setup_times[name] = float(duration)

    def mark_train_loop_start(self):
        self.train_loop_start_time = time.perf_counter()

    def mark_train_loop_end(self):
        self.train_loop_end_time = time.perf_counter()

    def record_step(self, data_time: float, model_time: float):
        self.data_times.append(float(data_time))
        self.model_times.append(float(model_time))
        self.step_times.append(float(data_time + model_time))

    def _collect_metrics(self, completed_steps: int):
        now = time.perf_counter()
        total_wall_time = now - self.start_time
        train_loop_end = self.train_loop_end_time or now
        train_loop_wall_time = (
            train_loop_end - self.train_loop_start_time if self.train_loop_start_time is not None else 0.0
        )
        measured_steps = len(self.step_times)
        measured_samples = measured_steps * self.per_loop_batch_size
        measured_step_time = sum(self.step_times)

        metrics = {
            "steps": {
                "completed": int(completed_steps),
                "measured": measured_steps,
            },
            "time_sec": {
                "total_wall": total_wall_time,
                "train_loop_wall": train_loop_wall_time,
                "measured_step": measured_step_time,
                "avg_data": _mean(self.data_times),
                "avg_model": _mean(self.model_times),
                "avg_step": _mean(self.step_times),
                "max_data": _max(self.data_times),
                "max_model": _max(self.model_times),
                "max_step": _max(self.step_times),
            },
            "throughput": {
                "samples_per_sec": measured_samples / measured_step_time if measured_step_time > 0 else 0.0,
                "steps_per_sec": measured_steps / measured_step_time if measured_step_time > 0 else 0.0,
                "wall_samples_per_sec": measured_samples / total_wall_time if total_wall_time > 0 else 0.0,
            },
            "data": {
                "wait_ratio": (sum(self.data_times) / measured_step_time) if measured_step_time > 0 else 0.0,
            },
            "batch": {
                "world_size": self.world_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "per_loop_batch_size": self.per_loop_batch_size,
                "total_batch_size": self.total_batch_size,
            },
            "setup_time_sec": dict(self.setup_times),
        }

        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = {
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                "max_reserved": torch.cuda.max_memory_reserved() / 1024**3,
            }

        return metrics

    def to_json_dict(self, completed_steps: int):
        return self._collect_metrics(completed_steps)

    def _flatten_metrics(self, metrics: dict):
        summary = {
            "completed_steps": metrics["steps"]["completed"],
            "measured_steps": metrics["steps"]["measured"],
            "total_wall_time_sec": metrics["time_sec"]["total_wall"],
            "train_loop_wall_time_sec": metrics["time_sec"]["train_loop_wall"],
            "measured_step_time_sec": metrics["time_sec"]["measured_step"],
            "avg_data_time_sec": metrics["time_sec"]["avg_data"],
            "avg_model_time_sec": metrics["time_sec"]["avg_model"],
            "avg_step_time_sec": metrics["time_sec"]["avg_step"],
            "max_data_time_sec": metrics["time_sec"]["max_data"],
            "max_model_time_sec": metrics["time_sec"]["max_model"],
            "max_step_time_sec": metrics["time_sec"]["max_step"],
            "samples_per_sec": metrics["throughput"]["samples_per_sec"],
            "steps_per_sec": metrics["throughput"]["steps_per_sec"],
            "wall_samples_per_sec": metrics["throughput"]["wall_samples_per_sec"],
            "data_wait_ratio": metrics["data"]["wait_ratio"],
            "world_size": metrics["batch"]["world_size"],
            "gradient_accumulation_steps": metrics["batch"]["gradient_accumulation_steps"],
            "per_loop_batch_size": metrics["batch"]["per_loop_batch_size"],
            "total_batch_size": metrics["batch"]["total_batch_size"],
        }

        for name, duration in metrics["setup_time_sec"].items():
            summary[f"setup/{name}_time_sec"] = duration

        if "gpu_memory_gb" in metrics:
            summary["max_gpu_memory_allocated_gb"] = metrics["gpu_memory_gb"]["max_allocated"]
            summary["max_gpu_memory_reserved_gb"] = metrics["gpu_memory_gb"]["max_reserved"]

        return summary

    def to_flat_dict(self, completed_steps: int):
        return self._flatten_metrics(self._collect_metrics(completed_steps))

    def to_dict(self, completed_steps: int):
        return self.to_flat_dict(completed_steps)

    def write(self, completed_steps: int):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_summary = self.to_json_dict(completed_steps)
        csv_summary = self._flatten_metrics(json_summary)
        json_path = self.output_dir / f"{self.output_basename}.json"
        csv_path = self.output_dir / f"{self.output_basename}.csv"

        with json_path.open("w") as f:
            json.dump(json_summary, f, indent=2)

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_summary.keys()))
            writer.writeheader()
            writer.writerow(csv_summary)

        return json_path, csv_path

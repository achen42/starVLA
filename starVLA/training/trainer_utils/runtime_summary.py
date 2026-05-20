import json
import time
from pathlib import Path

import torch


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _max(values):
    return max(values) if values else 0.0


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
    ):
        self.output_dir = Path(output_dir)
        self.per_loop_batch_size = int(per_loop_batch_size)
        self.total_batch_size = int(total_batch_size)
        self.world_size = int(world_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.start_time = time.perf_counter()
        self.data_times = []
        self.model_times = []
        self.step_times = []

    def record_step(self, data_time: float, model_time: float):
        self.data_times.append(float(data_time))
        self.model_times.append(float(model_time))
        self.step_times.append(float(data_time + model_time))

    def to_dict(self, completed_steps: int):
        total_wall_time = time.perf_counter() - self.start_time
        measured_steps = len(self.step_times)
        measured_samples = measured_steps * self.per_loop_batch_size
        measured_step_time = sum(self.step_times)

        summary = {
            "completed_steps": int(completed_steps),
            "measured_steps": measured_steps,
            "total_wall_time_sec": total_wall_time,
            "measured_step_time_sec": measured_step_time,
            "avg_data_time_sec": _mean(self.data_times),
            "avg_model_time_sec": _mean(self.model_times),
            "avg_step_time_sec": _mean(self.step_times),
            "max_data_time_sec": _max(self.data_times),
            "max_model_time_sec": _max(self.model_times),
            "max_step_time_sec": _max(self.step_times),
            "samples_per_sec": measured_samples / measured_step_time if measured_step_time > 0 else 0.0,
            "data_wait_ratio": (sum(self.data_times) / sum(self.step_times)) if sum(self.step_times) > 0 else 0.0,
            "world_size": self.world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "per_loop_batch_size": self.per_loop_batch_size,
            "total_batch_size": self.total_batch_size,
        }

        if torch.cuda.is_available():
            summary["max_gpu_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            summary["max_gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3

        return summary

    def write(self, completed_steps: int):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "runtime_summary.json"
        with out_path.open("w") as f:
            json.dump(self.to_dict(completed_steps), f, indent=2)
        return out_path

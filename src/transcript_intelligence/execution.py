from enum import StrEnum
from pathlib import Path
import shutil

from transcript_intelligence.io_utils import read_json, write_json
from transcript_intelligence.logging_setup import get_logger

log = get_logger(__name__)


class StageStatus(StrEnum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


STAGES = (
    "ingest",
    "privacy",
    "classify",
    "turns",
    "segments",
    "embeddings",
    "clustering",
    "topic_labels",
    "sentiment",
    "aggregation",
    "analytics",
)

# Pipeline stage name -> output directory stems under execution_<id>/
STAGE_OUTPUT_DIRS = {
    "ingest": ("ingest",),
    "privacy": ("privacy",),
    "classify": ("classify",),
    "turns": ("turns",),
    "segments": ("segments",),
    "embeddings": ("embeddings",),
    "clustering": ("clustering", "topic_representation"),
    "topic_labels": ("topic_label",),
    "sentiment": ("sentiment",),
    "aggregation": ("aggregation",),
    "analytics": ("analytical",),
}


class Execution:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"
        self.state = (
            read_json(self.state_path)
            if self.state_path.exists()
            else {
                "status": "running",
                "stages": {stage: StageStatus.pending.value for stage in STAGES},
            }
        )

    @classmethod
    def allocate(cls, output_directory: Path) -> "Execution":
        output_directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            path
            for path in output_directory.glob("execution_*")
            if path.is_dir()
        )
        if existing:
            latest = cls(existing[-1])
            if latest.state.get("status") != "complete":
                log.info(
                    "resuming incomplete execution",
                    path=str(latest.directory),
                )
                return latest
            next_id = int(existing[-1].name.split("_")[1]) + 1
        else:
            next_id = 0
        directory = output_directory / f"execution_{next_id}"
        directory.mkdir(parents=True, exist_ok=False)
        execution = cls(directory)
        execution.save()
        log.info("created new execution", path=str(directory))
        return execution

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def stage_dir(self, name: str) -> Path:
        path = self.directory / f"{name}_stage"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_stage_outputs(self, stage: str) -> None:
        for name in STAGE_OUTPUT_DIRS[stage]:
            path = self.directory / f"{name}_stage"
            if path.exists():
                shutil.rmtree(path)
                log.info(
                    "cleared stage output directory",
                    stage=stage,
                    path=str(path),
                )

    def is_complete(self, stage: str) -> bool:
        return self.state["stages"].get(stage) == StageStatus.complete.value

    def mark(self, stage: str, status: StageStatus) -> None:
        self.state["stages"][stage] = status.value
        if status == StageStatus.failed:
            self.state["status"] = "failed"
        self.save()

    def mark_complete(self) -> None:
        self.state["status"] = "complete"
        self.save()

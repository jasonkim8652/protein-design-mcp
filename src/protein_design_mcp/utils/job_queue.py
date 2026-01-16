"""
Job queue and status tracking for long-running design operations.

Provides a simple file-based job queue for tracking design jobs.
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Status of a design job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobProgress:
    """Progress information for a running job."""

    current_step: str  # "rfdiffusion", "proteinmpnn", "esmfold"
    designs_completed: int
    total_designs: int

    @property
    def percent_complete(self) -> float:
        """Calculate percent complete."""
        if self.total_designs == 0:
            return 0.0
        return (self.designs_completed / self.total_designs) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_step": self.current_step,
            "designs_completed": self.designs_completed,
            "total_designs": self.total_designs,
            "percent_complete": self.percent_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobProgress":
        """Create from dictionary."""
        return cls(
            current_step=data["current_step"],
            designs_completed=data["designs_completed"],
            total_designs=data["total_designs"],
        )


@dataclass
class JobInfo:
    """Information about a design job."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: JobProgress | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.to_dict() if self.progress else None,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobInfo":
        """Create from dictionary."""
        progress = None
        if data.get("progress"):
            progress = JobProgress.from_dict(data["progress"])

        return cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=progress,
            result=data.get("result"),
            error=data.get("error"),
        )


class JobQueue:
    """
    File-based job queue for tracking design operations.

    Jobs are stored as JSON files in a storage directory for persistence.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        """
        Initialize job queue.

        Args:
            storage_dir: Directory for storing job files. Defaults to ~/.cache/protein-design-mcp/jobs
        """
        if storage_dir is None:
            storage_dir = Path(os.environ.get(
                "CACHE_DIR",
                "~/.cache/protein-design-mcp"
            )).expanduser() / "jobs"
        else:
            storage_dir = Path(storage_dir)

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        """Get path to job file."""
        return self.storage_dir / f"{job_id}.json"

    def _save_job(self, job: JobInfo) -> None:
        """Save job to file."""
        job_file = self._job_file(job.job_id)
        with open(job_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

    def _load_job(self, job_id: str) -> JobInfo | None:
        """Load job from file."""
        job_file = self._job_file(job_id)
        if not job_file.exists():
            return None

        try:
            with open(job_file) as f:
                data = json.load(f)
            return JobInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def create_job(self) -> str:
        """
        Create a new job and return its ID.

        Returns:
            Unique job ID
        """
        job_id = str(uuid.uuid4())[:8]
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
        )
        self._save_job(job)
        return job_id

    def get_job(self, job_id: str) -> JobInfo | None:
        """
        Get job by ID.

        Args:
            job_id: Job ID to look up

        Returns:
            JobInfo or None if not found
        """
        return self._load_job(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status.

        Args:
            job_id: Job ID to update
            status: New status
        """
        job = self._load_job(job_id)
        if job is None:
            return

        job.status = status
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.now()

        self._save_job(job)

    def update_progress(self, job_id: str, progress: JobProgress) -> None:
        """
        Update job progress.

        Args:
            job_id: Job ID to update
            progress: New progress information
        """
        job = self._load_job(job_id)
        if job is None:
            return

        job.progress = progress
        self._save_job(job)

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """
        Mark job as completed with result.

        Args:
            job_id: Job ID to complete
            result: Job result data
        """
        job = self._load_job(job_id)
        if job is None:
            return

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.result = result
        self._save_job(job)

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Mark job as failed with error.

        Args:
            job_id: Job ID to fail
            error: Error message
        """
        job = self._load_job(job_id)
        if job is None:
            return

        job.status = JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error = error
        self._save_job(job)

    def list_jobs(self, status: JobStatus | None = None) -> list[JobInfo]:
        """
        List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of jobs matching criteria
        """
        jobs = []
        for job_file in self.storage_dir.glob("*.json"):
            job_id = job_file.stem
            job = self._load_job(job_id)
            if job is not None:
                if status is None or job.status == status:
                    jobs.append(job)

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs


# Global job queue instance
_job_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def set_job_queue(queue: JobQueue) -> None:
    """Set the global job queue instance (for testing)."""
    global _job_queue
    _job_queue = queue

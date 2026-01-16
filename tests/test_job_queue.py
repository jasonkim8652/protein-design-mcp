"""Tests for job queue and status tracking."""

import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protein_design_mcp.utils.job_queue import (
    JobQueue,
    JobStatus,
    JobInfo,
    JobProgress,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """JobStatus should have expected values."""
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_job_info_creation(self):
        """Should create JobInfo with all fields."""
        job = JobInfo(
            job_id="test_123",
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
        )
        assert job.job_id == "test_123"
        assert job.status == JobStatus.QUEUED
        assert job.progress is None
        assert job.result is None
        assert job.error is None

    def test_job_info_with_progress(self):
        """Should store progress information."""
        progress = JobProgress(
            current_step="rfdiffusion",
            designs_completed=3,
            total_designs=10,
        )
        job = JobInfo(
            job_id="test_456",
            status=JobStatus.RUNNING,
            created_at=datetime.now(),
            progress=progress,
        )
        assert job.progress.current_step == "rfdiffusion"
        assert job.progress.designs_completed == 3


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_job_progress_fields(self):
        """Should have expected fields."""
        progress = JobProgress(
            current_step="proteinmpnn",
            designs_completed=5,
            total_designs=10,
        )
        assert progress.current_step == "proteinmpnn"
        assert progress.designs_completed == 5
        assert progress.total_designs == 10

    def test_job_progress_percent_complete(self):
        """Should calculate percent complete."""
        progress = JobProgress(
            current_step="esmfold",
            designs_completed=7,
            total_designs=10,
        )
        assert progress.percent_complete == 70.0

    def test_job_progress_zero_total(self):
        """Should handle zero total gracefully."""
        progress = JobProgress(
            current_step="rfdiffusion",
            designs_completed=0,
            total_designs=0,
        )
        assert progress.percent_complete == 0.0


class TestJobQueue:
    """Tests for JobQueue class."""

    def test_queue_init(self, tmp_path):
        """Should initialize empty queue."""
        queue = JobQueue(storage_dir=tmp_path)
        assert len(queue.list_jobs()) == 0

    def test_queue_create_job(self, tmp_path):
        """Should create new job with unique ID."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        assert job_id is not None
        assert len(job_id) > 0

    def test_queue_create_job_unique_ids(self, tmp_path):
        """Should generate unique IDs for each job."""
        queue = JobQueue(storage_dir=tmp_path)
        ids = [queue.create_job() for _ in range(10)]

        assert len(set(ids)) == 10  # All unique

    def test_queue_get_job(self, tmp_path):
        """Should retrieve job by ID."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        job = queue.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.status == JobStatus.QUEUED

    def test_queue_get_nonexistent_job(self, tmp_path):
        """Should return None for nonexistent job."""
        queue = JobQueue(storage_dir=tmp_path)
        job = queue.get_job("nonexistent_id")
        assert job is None

    def test_queue_update_status(self, tmp_path):
        """Should update job status."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        queue.update_status(job_id, JobStatus.RUNNING)
        job = queue.get_job(job_id)
        assert job.status == JobStatus.RUNNING

    def test_queue_update_progress(self, tmp_path):
        """Should update job progress."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        progress = JobProgress(
            current_step="rfdiffusion",
            designs_completed=2,
            total_designs=10,
        )
        queue.update_progress(job_id, progress)

        job = queue.get_job(job_id)
        assert job.progress is not None
        assert job.progress.designs_completed == 2

    def test_queue_complete_job(self, tmp_path):
        """Should mark job as completed with result."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        result = {"designs": [], "summary": {"total": 10}}
        queue.complete_job(job_id, result)

        job = queue.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result == result

    def test_queue_fail_job(self, tmp_path):
        """Should mark job as failed with error."""
        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        queue.fail_job(job_id, "Out of memory")

        job = queue.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "memory" in job.error.lower()

    def test_queue_list_jobs(self, tmp_path):
        """Should list all jobs."""
        queue = JobQueue(storage_dir=tmp_path)
        id1 = queue.create_job()
        id2 = queue.create_job()
        id3 = queue.create_job()

        jobs = queue.list_jobs()
        assert len(jobs) == 3

    def test_queue_list_jobs_by_status(self, tmp_path):
        """Should filter jobs by status."""
        queue = JobQueue(storage_dir=tmp_path)
        id1 = queue.create_job()
        id2 = queue.create_job()
        id3 = queue.create_job()

        queue.update_status(id1, JobStatus.RUNNING)
        queue.update_status(id2, JobStatus.COMPLETED)

        running = queue.list_jobs(status=JobStatus.RUNNING)
        assert len(running) == 1
        assert running[0].job_id == id1

    def test_queue_persistence(self, tmp_path):
        """Jobs should persist across queue instances."""
        queue1 = JobQueue(storage_dir=tmp_path)
        job_id = queue1.create_job()
        queue1.update_status(job_id, JobStatus.RUNNING)

        # Create new queue instance
        queue2 = JobQueue(storage_dir=tmp_path)
        job = queue2.get_job(job_id)

        assert job is not None
        assert job.status == JobStatus.RUNNING


class TestGetDesignStatus:
    """Tests for get_design_status tool."""

    @pytest.mark.asyncio
    async def test_get_status_returns_dict(self, tmp_path):
        """get_design_status should return status dictionary."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            result = await get_design_status(job_id=job_id)

        assert isinstance(result, dict)
        assert "status" in result

    @pytest.mark.asyncio
    async def test_get_status_queued(self, tmp_path):
        """Should return queued status for new job."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            result = await get_design_status(job_id=job_id)

        assert result["status"] == "queued"

    @pytest.mark.asyncio
    async def test_get_status_running_with_progress(self, tmp_path):
        """Should return progress for running job."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, JobProgress(
            current_step="proteinmpnn",
            designs_completed=5,
            total_designs=10,
        ))

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            result = await get_design_status(job_id=job_id)

        assert result["status"] == "running"
        assert result["progress"]["current_step"] == "proteinmpnn"
        assert result["progress"]["designs_completed"] == 5

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, tmp_path):
        """Should raise error for nonexistent job."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            with pytest.raises(ValueError, match="[Jj]ob.*not found"):
                await get_design_status(job_id="nonexistent")

    @pytest.mark.asyncio
    async def test_get_status_completed_with_result(self, tmp_path):
        """Should include result info for completed job."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()
        result_data = {"designs": [{"id": "d1"}], "summary": {"total": 1}}
        queue.complete_job(job_id, result_data)

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            result = await get_design_status(job_id=job_id)

        assert result["status"] == "completed"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_status_failed_with_error(self, tmp_path):
        """Should include error for failed job."""
        from protein_design_mcp.tools.status import get_design_status

        queue = JobQueue(storage_dir=tmp_path)
        job_id = queue.create_job()
        queue.fail_job(job_id, "GPU memory error")

        with patch("protein_design_mcp.tools.status.get_job_queue", return_value=queue):
            result = await get_design_status(job_id=job_id)

        assert result["status"] == "failed"
        assert "error" in result
        assert "GPU" in result["error"]

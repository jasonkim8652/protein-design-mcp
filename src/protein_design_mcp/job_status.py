"""
Design status tool - check status of running design jobs.

This tool allows checking the status and progress of long-running
design operations.
"""

from typing import Any

from protein_design_mcp.utils.job_queue import get_job_queue, JobStatus


async def get_design_status(job_id: str) -> dict[str, Any]:
    """
    Get the status of a design job.

    Args:
        job_id: ID of the job to check

    Returns:
        Dictionary containing:
        - status: "queued", "running", "completed", or "failed"
        - progress: Progress info for running jobs
        - result: Result summary for completed jobs
        - error: Error message for failed jobs
        - estimated_time_remaining: Estimated time to completion

    Raises:
        ValueError: If job_id is not found
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if job is None:
        raise ValueError(f"Job not found: {job_id}")

    result: dict[str, Any] = {
        "status": job.status.value,
        "job_id": job.job_id,
        "created_at": job.created_at.isoformat(),
    }

    # Add progress for running jobs
    if job.progress is not None:
        result["progress"] = {
            "current_step": job.progress.current_step,
            "designs_completed": job.progress.designs_completed,
            "total_designs": job.progress.total_designs,
            "percent_complete": job.progress.percent_complete,
        }

        # Estimate remaining time based on progress
        result["estimated_time_remaining"] = _estimate_time_remaining(
            job.progress.designs_completed,
            job.progress.total_designs,
            job.progress.current_step,
        )

    # Add result for completed jobs
    if job.status == JobStatus.COMPLETED and job.result is not None:
        result["result"] = job.result
        if job.completed_at:
            result["completed_at"] = job.completed_at.isoformat()

    # Add error for failed jobs
    if job.status == JobStatus.FAILED and job.error is not None:
        result["error"] = job.error
        if job.completed_at:
            result["completed_at"] = job.completed_at.isoformat()

    return result


def _estimate_time_remaining(
    completed: int,
    total: int,
    current_step: str,
) -> str | None:
    """
    Estimate remaining time based on progress.

    This is a rough estimate based on typical pipeline timings:
    - RFdiffusion: ~2-5 min per design
    - ProteinMPNN: ~30 sec per backbone
    - ESMFold: ~1-2 min per sequence

    Args:
        completed: Number of designs completed
        total: Total number of designs
        current_step: Current pipeline step

    Returns:
        Human-readable time estimate or None if cannot estimate
    """
    if total == 0:
        return None

    remaining = total - completed

    # Rough time per design in minutes based on step
    step_times = {
        "rfdiffusion": 3.0,  # minutes per design
        "proteinmpnn": 0.5,
        "esmfold": 1.5,
    }

    # Estimate based on current step
    time_per = step_times.get(current_step, 2.0)
    minutes_remaining = remaining * time_per

    if minutes_remaining < 1:
        return "less than 1 minute"
    elif minutes_remaining < 60:
        return f"{int(minutes_remaining)} minutes"
    else:
        hours = minutes_remaining / 60
        return f"{hours:.1f} hours"

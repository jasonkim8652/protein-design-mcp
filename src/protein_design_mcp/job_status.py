"""
Design status tool - check status of running design jobs.

This tool allows checking the status and progress of long-running
design operations.
"""

from datetime import datetime
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
            job.started_at,
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
    started_at: datetime | None,
) -> str | None:
    """
    Estimate remaining time from THIS job's own observed progress rate.

    The previous implementation hardcoded per-minute costs for
    "rfdiffusion"/"proteinmpnn"/"esmfold" -- the three steps of the OLD
    composite design pipeline. That pipeline no longer exists: every
    generative/structure-prediction tool this server registers today
    (run_rfdiffusion_binder, run_boltz, run_alphafold3, ...) is a single
    engine invoked directly, not a step name out of that fixed set of
    three, and there is no live data on this host calibrating what those
    three numbers should even be for the CURRENT tool roster. Shipping a
    stale per-step table under a new name would be exactly the kind of
    fabricated timing WAVE-COMMON warns against -- confidently wrong is
    worse than admittedly unknown.

    Instead, the estimate is derived from THIS job's own progress so far:
    (elapsed wall-clock time since ``started_at``) / (designs completed),
    projected across the remaining designs. This adapts automatically to
    whichever engine is actually running, at the cost of being unavailable
    until at least one design has completed.

    Returns None whenever a rate cannot be computed honestly:
    - ``total`` is 0 (nothing to estimate against),
    - ``started_at`` is unset (no elapsed time to measure a rate from), or
    - ``completed`` is 0 (explicitly checked, not just falsy-skipped --
      zero designs completed in *some* elapsed time is not a rate, it is a
      job that has not produced its first result yet).

    Args:
        completed: Number of designs completed so far.
        total: Total number of designs the job is producing.
        started_at: When the job began running, or None if it has not
            started.

    Returns:
        Human-readable time estimate, or None if it cannot be computed.
    """
    if total == 0 or started_at is None or completed <= 0:
        return None

    remaining = total - completed
    if remaining <= 0:
        return "less than 1 minute"

    elapsed_minutes = (datetime.now() - started_at).total_seconds() / 60.0
    if elapsed_minutes <= 0:
        # Clock skew or a job "started" in the future -- refuse to project
        # a rate from a non-positive elapsed time rather than divide by
        # (near) zero and return a nonsense estimate.
        return None

    minutes_per_design = elapsed_minutes / completed
    minutes_remaining = minutes_per_design * remaining

    if minutes_remaining < 1:
        return "less than 1 minute"
    elif minutes_remaining < 60:
        return f"{int(minutes_remaining)} minutes"
    else:
        hours = minutes_remaining / 60
        return f"{hours:.1f} hours"

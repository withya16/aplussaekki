"""Job 서비스"""
from app.storage.job_store import JobStore
from app.core.errors import JobNotFoundError


class JobService:
    """Job 관리 서비스"""
    
    @staticmethod
    def get_job_status(job_id: str) -> "JobStatus":
        """Job 상태 조회"""
        from app.models.job import JobStatus
        
        job = JobStore.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        return job



"""
Model training and evaluation routes
"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.job import Job
from app.tasks.tasks_training import train_ppi_model_task, train_drug_matcher_task
from app.core.blockchain import log_blockchain_event

router = APIRouter()

class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    split_seed: int = 42
    validation_split: float = 0.2

class TrainingRequest(BaseModel):
    config: TrainingConfig
    csv_path: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: str

class MetricsResponse(BaseModel):
    roc_auc: float
    pr_auc: float
    accuracy: float
    loss: float
    confusion_matrix_url: str
    loss_curve_url: str
    training_config: dict
    dataset_info: dict

@router.post("/train/ppi", response_model=JobResponse)
async def train_ppi_model(
    request: TrainingRequest,
    csv_file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Train PPI prediction model"""
    
    if current_user.role not in ["admin", "researcher"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Handle CSV upload
    csv_path = request.csv_path
    if csv_file:
        # Save uploaded file
        import aiofiles
        import os
        
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        csv_path = f"{upload_dir}/{csv_file.filename}"
        
        async with aiofiles.open(csv_path, 'wb') as f:
            content = await csv_file.read()
            await f.write(content)
    
    if not csv_path:
        raise HTTPException(status_code=400, detail="CSV file or path required")
    
    # Create job record
    job = Job(
        user_id=current_user.id,
        job_type="train_ppi",
        status="pending",
        progress=0.0,
        config=request.config.dict()
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start training task
    task = train_ppi_model_task.delay(
        job_id=job.id,
        csv_path=csv_path,
        config=request.config.dict(),
        user_id=current_user.id
    )
    
    job.task_id = task.id
    await db.commit()
    
    # Log to blockchain
    await log_blockchain_event(
        user_address=str(current_user.id),
        action="models.train",
        ref_id=str(job.id)
    )
    
    return JobResponse(
        job_id=task.id,
        status="pending",
        progress=0.0,
        created_at=job.created_at.isoformat()
    )

@router.post("/train/drug", response_model=JobResponse)
async def train_drug_matcher(
    config: TrainingConfig,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Train drug-protein matcher"""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Create job record
    job = Job(
        user_id=current_user.id,
        job_type="train_drug",
        status="pending",
        progress=0.0,
        config=config.dict()
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start training task
    task = train_drug_matcher_task.delay(
        job_id=job.id,
        config=config.dict(),
        user_id=current_user.id
    )
    
    job.task_id = task.id
    await db.commit()
    
    # Log to blockchain
    await log_blockchain_event(
        user_address=str(current_user.id),
        action="models.train.drug",
        ref_id=str(job.id)
    )
    
    return JobResponse(
        job_id=task.id,
        status="pending",
        progress=0.0,
        created_at=job.created_at.isoformat()
    )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get training job status"""
    from sqlalchemy import select
    
    result = await db.execute(
        select(Job).where(Job.task_id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobResponse(
        job_id=job.task_id,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at.isoformat()
    )

@router.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics(
    model_type: str = "ppi",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get latest model metrics"""
    
    try:
        if model_type == "ppi":
            from app.ml.train.metrics import get_ppi_metrics
            metrics = await get_ppi_metrics()
        elif model_type == "drug":
            from app.ml.train.metrics import get_drug_metrics
            metrics = await get_drug_metrics()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return MetricsResponse(
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            accuracy=metrics["accuracy"],
            loss=metrics["loss"],
            confusion_matrix_url=metrics["confusion_matrix_url"],
            loss_curve_url=metrics["loss_curve_url"],
            training_config=metrics["config"],
            dataset_info=metrics["dataset_info"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

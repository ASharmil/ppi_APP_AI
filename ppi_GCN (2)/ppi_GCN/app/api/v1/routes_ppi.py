"""
Protein-Protein Interaction prediction routes
"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.prediction import Prediction
from app.tasks.tasks_prediction import predict_ppi_task
from app.services.uniprot_client import get_protein_sequence
from app.services.pdb_client import get_protein_structure
from app.core.blockchain import log_blockchain_event

router = APIRouter()

class PPIPredictionRequest(BaseModel):
    protein_a: str  # UniProt ID or sequence
    protein_b: str  # UniProt ID or sequence
    prediction_type: str = "binding_affinity"

class ResidueAnnotation(BaseModel):
    position: int
    amino_acid: str
    interaction_probability: float
    binding_site: bool

class PPIPredictionResponse(BaseModel):
    prediction_id: int
    binding_affinity: float
    docking_score: float
    confidence: float
    residue_annotations: List[ResidueAnnotation]
    structure_urls: dict
    tx_hash: str

@router.post("/predict", response_model=PPIPredictionResponse)
async def predict_ppi(
    request: PPIPredictionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Predict protein-protein interaction"""
    
    # Fetch protein sequences if IDs provided
    if len(request.protein_a) < 20:  # Likely a UniProt ID
        seq_a = await get_protein_sequence(request.protein_a)
    else:
        seq_a = request.protein_a
    
    if len(request.protein_b) < 20:  # Likely a UniProt ID
        seq_b = await get_protein_sequence(request.protein_b)
    else:
        seq_b = request.protein_b
    
    # Create prediction record
    prediction = Prediction(
        user_id=current_user.id,
        protein_a_id=request.protein_a,
        protein_b_id=request.protein_b,
        protein_a_sequence=seq_a,
        protein_b_sequence=seq_b,
        status="pending"
    )
    
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    
    # Start prediction task
    task = predict_ppi_task.delay(
        prediction_id=prediction.id,
        protein_a_seq=seq_a,
        protein_b_seq=seq_b,
        user_id=current_user.id
    )
    
    # Update prediction with task ID
    prediction.task_id = task.id
    await db.commit()
    
    # Log to blockchain
    tx_hash = await log_blockchain_event(
        user_address=str(current_user.id),
        action="ppi.predict",
        ref_id=str(prediction.id)
    )
    
    prediction.tx_hash = tx_hash
    await db.commit()
    
    return PPIPredictionResponse(
        prediction_id=prediction.id,
        binding_affinity=0.0,  # Will be updated by background task
        docking_score=0.0,
        confidence=0.0,
        residue_annotations=[],
        structure_urls={},
        tx_hash=tx_hash
    )

@router.get("/predictions/{prediction_id}", response_model=PPIPredictionResponse)
async def get_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get prediction results"""
    prediction = await db.get(Prediction, prediction_id)
    
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )
    
    if prediction.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return PPIPredictionResponse(
        prediction_id=prediction.id,
        binding_affinity=prediction.binding_affinity or 0.0,
        docking_score=prediction.docking_score or 0.0,
        confidence=prediction.confidence or 0.0,
        residue_annotations=prediction.residue_annotations or [],
        structure_urls=prediction.structure_urls or {},
        tx_hash=prediction.tx_hash or ""
    )

@router.get("/predictions", response_model=List[PPIPredictionResponse])
async def list_predictions(
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List user's predictions"""
    from sqlalchemy import select
    
    query = select(Prediction).where(Prediction.user_id == current_user.id)
    if current_user.role == "admin":
        query = select(Prediction)
    
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    predictions = result.scalars().all()
    
    return [
        PPIPredictionResponse(
            prediction_id=p.id,
            binding_affinity=p.binding_affinity or 0.0,
            docking_score=p.docking_score or 0.0,
            confidence=p.confidence or 0.0,
            residue_annotations=p.residue_annotations or [],
            structure_urls=p.structure_urls or {},
            tx_hash=p.tx_hash or ""
        )
        for p in predictions
    ]
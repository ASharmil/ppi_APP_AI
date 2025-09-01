"""
Drug discovery routes
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.drug import Drug
from app.tasks.tasks_drug_sync import sync_drug_databases_task
from app.tasks.tasks_training import train_drug_matcher_task
from app.core.blockchain import log_blockchain_event

router = APIRouter()

class DrugCandidate(BaseModel):
    drug_id: str
    drug_name: str
    source: str  # drugbank, chembl, pubchem
    confidence: float
    binding_score: float
    properties: dict

class DrugSuggestionRequest(BaseModel):
    protein_id: Optional[str] = None
    protein_sequence: Optional[str] = None
    malfunction_description: Optional[str] = None
    top_k: int = 10

class DrugSuggestionResponse(BaseModel):
    protein_id: str
    candidates: List[DrugCandidate]
    tx_hash: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float

@router.post("/suggest", response_model=DrugSuggestionResponse)
async def suggest_drugs(
    request: DrugSuggestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Suggest drug candidates for a protein target"""
    
    if not request.protein_id and not request.protein_sequence:
        raise HTTPException(
            status_code=400,
            detail="Either protein_id or protein_sequence must be provided"
        )
    
    try:
        # Import here to avoid circular imports
        from app.ml.infer.predictor import DrugMatcher
        
        drug_matcher = DrugMatcher()
        
        # Get drug candidates
        candidates = await drug_matcher.suggest_drugs(
            protein_id=request.protein_id,
            protein_sequence=request.protein_sequence,
            malfunction_description=request.malfunction_description,
            top_k=request.top_k
        )
        
        # Log to blockchain
        tx_hash = await log_blockchain_event(
            user_address=str(current_user.id),
            action="drug.suggest",
            ref_id=request.protein_id or "sequence"
        )
        
        return DrugSuggestionResponse(
            protein_id=request.protein_id or "provided_sequence",
            candidates=[
                DrugCandidate(
                    drug_id=c["drug_id"],
                    drug_name=c["drug_name"],
                    source=c["source"],
                    confidence=c["confidence"],
                    binding_score=c["binding_score"],
                    properties=c["properties"]
                )
                for c in candidates
            ],
            tx_hash=tx_hash
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drug suggestion failed: {str(e)}"
        )

@router.post("/sync", response_model=JobResponse)
async def sync_drug_databases(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Sync drug databases (admin only)"""
    
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    # Start sync task
    task = sync_drug_databases_task.delay()
    
    # Log to blockchain
    await log_blockchain_event(
        user_address=str(current_user.id),
        action="drugs.sync",
        ref_id=task.id
    )
    
    return JobResponse(
        job_id=task.id,
        status="pending",
        progress=0.0
    )

@router.get("/databases/status")
async def get_database_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get drug database sync status"""
    from sqlalchemy import select, func
    
    # Get drug counts by source
    drugbank_count = await db.execute(
        select(func.count()).select_from(Drug).where(Drug.source == "drugbank")
    )
    chembl_count = await db.execute(
        select(func.count()).select_from(Drug).where(Drug.source == "chembl")
    )
    pubchem_count = await db.execute(
        select(func.count()).select_from(Drug).where(Drug.source == "pubchem")
    )
    
    return {
        "drugbank_drugs": drugbank_count.scalar(),
        "chembl_drugs": chembl_count.scalar(),
        "pubchem_drugs": pubchem_count.scalar(),
        "last_sync": None  # TODO: implement last sync tracking
    }

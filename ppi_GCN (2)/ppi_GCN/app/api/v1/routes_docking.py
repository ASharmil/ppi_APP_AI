"""
Molecular docking routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.ml.data.featurizers import HaddockFeatureBuilder
from app.ml.infer.predictor import DockingPredictor

router = APIRouter()

class HaddockFeatures(BaseModel):
    electrostatic_energy: float
    van_der_waals_energy: float
    desolvation_energy: float
    restraint_violation_energy: float
    buried_surface_area: float
    z_score: float

class DockingRequest(BaseModel):
    protein_a: str
    protein_b: str
    haddock_features: Optional[HaddockFeatures] = None

class DockingResponse(BaseModel):
    docking_score: float
    binding_affinity: float
    confidence: float
    energy_breakdown: dict

@router.post("/score", response_model=DockingResponse)
async def score_docking(
    request: DockingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Score protein docking using HADDOCK features"""
    
    try:
        # Build features if not provided
        if request.haddock_features is None:
            feature_builder = HaddockFeatureBuilder()
            features = await feature_builder.build_features(
                request.protein_a, 
                request.protein_b
            )
        else:
            features = request.haddock_features.dict()
        
        # Load docking predictor
        predictor = DockingPredictor()
        
        # Get prediction
        result = await predictor.predict_docking(
            protein_a=request.protein_a,
            protein_b=request.protein_b,
            haddock_features=features
        )
        
        return DockingResponse(
            docking_score=result["docking_score"],
            binding_affinity=result["binding_affinity"],
            confidence=result["confidence"],
            energy_breakdown=result["energy_breakdown"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Docking prediction failed: {str(e)}"
        )
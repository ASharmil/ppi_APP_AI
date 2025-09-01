"""
Institution management routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.institution import Institution

router = APIRouter()

class InstitutionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    address: Optional[str] = None
    contact_email: Optional[str] = None

class InstitutionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    address: Optional[str]
    contact_email: Optional[str]
    user_count: int
    prediction_count: int
    is_active: bool

@router.get("/", response_model=List[InstitutionResponse])
async def list_institutions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all institutions (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from sqlalchemy import select, func
    
    # Get institutions with user counts
    query = select(
        Institution,
        func.count(User.id).label("user_count")
    ).outerjoin(User).group_by(Institution.id)
    
    result = await db.execute(query)
    institutions = result.all()
    
    return [
        InstitutionResponse(
            id=inst.Institution.id,
            name=inst.Institution.name,
            description=inst.Institution.description,
            address=inst.Institution.address,
            contact_email=inst.Institution.contact_email,
            user_count=inst.user_count,
            prediction_count=0,  # TODO: implement prediction count
            is_active=inst.Institution.is_active
        )
        for inst in institutions
    ]

@router.post("/", response_model=InstitutionResponse)
async def create_institution(
    institution_data: InstitutionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create new institution (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    institution = Institution(
        name=institution_data.name,
        description=institution_data.description,
        address=institution_data.address,
        contact_email=institution_data.contact_email,
        is_active=True
    )
    
    db.add(institution)
    await db.commit()
    await db.refresh(institution)
    
    return InstitutionResponse(
        id=institution.id,
        name=institution.name,
        description=institution.description,
        address=institution.address,
        contact_email=institution.contact_email,
        user_count=0,
        prediction_count=0,
        is_active=institution.is_active
    )

@router.get("/{institution_id}", response_model=InstitutionResponse)
async def get_institution(
    institution_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get institution details"""
    if current_user.role != "admin" and current_user.institution_id != institution_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    institution = await db.get(Institution, institution_id)
    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")
    
    from sqlalchemy import select, func
    
    # Get user count
    user_count_result = await db.execute(
        select(func.count()).select_from(User).where(User.institution_id == institution_id)
    )
    user_count = user_count_result.scalar()
    
    return InstitutionResponse(
        id=institution.id,
        name=institution.name,
        description=institution.description,
        address=institution.address,
        contact_email=institution.contact_email,
        user_count=user_count,
        prediction_count=0,  # TODO: implement
        is_active=institution.is_active
    )
"""
Admin routes for blockchain and system management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.blockchain_event import BlockchainEvent
from app.core.blockchain import deploy_contract, get_contract_info

router = APIRouter()

class BlockchainEventResponse(BaseModel):
    id: int
    user_address: str
    action: str
    ref_id: str
    tx_hash: str
    timestamp: datetime
    block_number: Optional[int]

class ContractDeployResponse(BaseModel):
    contract_address: str
    tx_hash: str
    abi: dict

@router.get("/blockchain/events", response_model=List[BlockchainEventResponse])
async def get_blockchain_events(
    action_type: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    limit: int = Query(50, le=1000),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get blockchain events with filtering"""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from sqlalchemy import select
    
    query = select(BlockchainEvent)
    
    if action_type:
        query = query.where(BlockchainEvent.action.like(f"{action_type}%"))
    
    if user_id:
        query = query.where(BlockchainEvent.user_address == str(user_id))
    
    query = query.order_by(BlockchainEvent.timestamp.desc())
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    return [
        BlockchainEventResponse(
            id=event.id,
            user_address=event.user_address,
            action=event.action,
            ref_id=event.ref_id,
            tx_hash=event.tx_hash,
            timestamp=event.timestamp,
            block_number=event.block_number
        )
        for event in events
    ]

@router.post("/blockchain/deploy", response_model=ContractDeployResponse)
async def deploy_blockchain_contract(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deploy or redeploy blockchain contract (dev only)"""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await deploy_contract()
        return ContractDeployResponse(
            contract_address=result["address"],
            tx_hash=result["tx_hash"],
            abi=result["abi"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Contract deployment failed: {str(e)}"
        )

@router.get("/blockchain/contract")
async def get_contract_details(
    current_user: User = Depends(get_current_user)
):
    """Get current contract details"""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        contract_info = await get_contract_info()
        return contract_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get contract info: {str(e)}"
        )
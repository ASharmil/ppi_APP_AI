from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class BlockchainEvent(Base):
    __tablename__ = "blockchain_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    action = Column(String, nullable=False)
    ref_id = Column(String)
    tx_hash = Column(String, unique=True, index=True)
    block_number = Column(Integer)
    gas_used = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")
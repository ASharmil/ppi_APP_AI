from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    celery_task_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    job_type = Column(String)  # train_ppi, train_drug, sync_drugs, predict_ppi
    status = Column(String, default="pending")  # pending, running, success, failed
    progress = Column(Integer, default=0)  # 0-100
    
    config = Column(JSON)  # Training config, prediction params, etc.
    result = Column(JSON)  # Final results
    error_message = Column(Text)
    
    tx_hash = Column(String)  # Blockchain transaction hash
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")
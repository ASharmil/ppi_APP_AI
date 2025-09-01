from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    protein_a_id = Column(Integer, ForeignKey("proteins.id"))
    protein_b_id = Column(Integer, ForeignKey("proteins.id"))
    
    interaction_score = Column(Float)
    binding_affinity = Column(Float)
    confidence = Column(Float)
    
    residue_annotations = Column(JSON)  # Per-residue interaction probabilities
    haddock_features = Column(JSON)
    
    # File paths
    pdb_path = Column(String)
    cif_path = Column(String)
    gltf_path = Column(String)
    
    # Blockchain
    tx_hash = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="predictions")
    protein_a = relationship("Protein", foreign_keys=[protein_a_id])
    protein_b = relationship("Protein", foreign_keys=[protein_b_id])
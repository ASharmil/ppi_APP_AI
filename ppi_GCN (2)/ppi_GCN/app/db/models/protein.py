from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.db.base import Base

class Protein(Base):
    __tablename__ = "proteins"
    
    id = Column(Integer, primary_key=True, index=True)
    uniprot_id = Column(String, unique=True, index=True)
    pdb_id = Column(String, index=True)
    sequence = Column(Text)
    name = Column(String)
    organism = Column(String)
    function = Column(Text)
    structure_data = Column(JSON)  # PDB/mmCIF metadata
    cached_features = Column(JSON)  # Precomputed features
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float
from sqlalchemy.sql import func
from app.db.base import Base

class Drug(Base):
    __tablename__ = "drugs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Source identifiers
    drugbank_id = Column(String, index=True)
    chembl_id = Column(String, index=True)
    pubchem_cid = Column(String, index=True)
    
    name = Column(String, nullable=False)
    smiles = Column(String)
    inchi = Column(String)
    molecular_weight = Column(Float)
    
    # Properties
    properties = Column(JSON)  # RDKit descriptors, etc.
    targets = Column(JSON)  # Known protein targets
    indications = Column(JSON)  # Therapeutic uses
    
    source = Column(String)  # drugbank, chembl, pubchem
    sync_version = Column(Integer, default=1)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
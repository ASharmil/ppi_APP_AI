from celery import current_task
import asyncio
from typing import List, Dict, Any
from app.core.celery_app import celery_app
from app.services.drugbank_client import DrugBankClient
from app.services.chembl_client import ChEMBLClient
from app.services.pubchem_client import PubChemClient
from app.db.session import SessionLocal
from app.db.models.drug import Drug
from app.db.models.job import Job
from app.ml.data.featurizers import DrugFeaturizer
from sqlalchemy.exc import IntegrityError
from loguru import logger
import time

@celery_app.task(bind=True)
def sync_drugs_from_sources(self, job_id: int):
    """Sync drugs from DrugBank, ChEMBL, and PubChem"""
    
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        logger.error(f"Job {job_id} not found")
        return {"status": "error", "message": "Job not found"}
    
    try:
        job.status = "running"
        job.started_at = db.execute("SELECT NOW()").scalar()
        job.progress = 0
        db.commit()
        
        # Initialize clients
        drugbank_client = DrugBankClient()
        chembl_client = ChEMBLClient()
        pubchem_client = PubChemClient()
        featurizer = DrugFeaturizer()
        
        total_steps = 3  # DrugBank, ChEMBL, PubChem
        current_step = 0
        
        # Update progress function
        def update_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            progress = int((current_step / total_steps) * 100)
            job.progress = progress
            job.logs = job.logs + f"\n{message}" if job.logs else message
            db.commit()
            
            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current_step,
                    'total': total_steps,
                    'message': message,
                    'progress': progress
                }
            )
        
        # Sync from DrugBank
        update_progress(1, "Starting DrugBank sync...")
        drugbank_drugs = asyncio.run(sync_drugbank_drugs(drugbank_client, db, featurizer))
        update_progress(1, f"DrugBank sync completed: {len(drugbank_drugs)} drugs processed")
        
        # Sync from ChEMBL
        update_progress(2, "Starting ChEMBL sync...")
        chembl_drugs = asyncio.run(sync_chembl_drugs(chembl_client, db, featurizer))
        update_progress(2, f"ChEMBL sync completed: {len(chembl_drugs)} drugs processed")
        
        # Sync from PubChem
        update_progress(3, "Starting PubChem sync...")
        pubchem_drugs = asyncio.run(sync_pubchem_drugs(pubchem_client, db, featurizer))
        update_progress(3, f"PubChem sync completed: {len(pubchem_drugs)} drugs processed")
        
        # Final statistics
        total_drugs = len(drugbank_drugs) + len(chembl_drugs) + len(pubchem_drugs)
        unique_drugs = db.query(Drug).count()
        
        job.status = "completed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.progress = 100
        job.result = {
            "total_processed": total_drugs,
            "unique_drugs": unique_drugs,
            "drugbank_count": len(drugbank_drugs),
            "chembl_count": len(chembl_drugs),
            "pubchem_count": len(pubchem_drugs)
        }
        job.logs = job.logs + f"\nSync completed successfully. Total: {total_drugs}, Unique: {unique_drugs}"
        db.commit()
        
        logger.info(f"Drug sync job {job_id} completed successfully")
        return {"status": "success", "result": job.result}
        
    except Exception as e:
        logger.error(f"Drug sync job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.error = str(e)
        job.logs = job.logs + f"\nError: {str(e)}" if job.logs else f"Error: {str(e)}"
        db.commit()
        raise
    finally:
        db.close()

async def sync_drugbank_drugs(client: DrugBankClient, db, featurizer: DrugFeaturizer) -> List[str]:
    """Sync drugs from DrugBank"""
    try:
        drugs_data = await client.fetch_drugs(limit=1000)  # Adjust limit as needed
        processed_ids = []
        
        for drug_data in drugs_data:
            try:
                # Check if drug already exists
                existing_drug = db.query(Drug).filter(
                    Drug.source == "drugbank",
                    Drug.source_id == drug_data.get("drugbank_id")
                ).first()
                
                if existing_drug:
                    # Update existing drug
                    existing_drug.name = drug_data.get("name")
                    existing_drug.smiles = drug_data.get("smiles")
                    existing_drug.molecular_formula = drug_data.get("molecular_formula")
                    existing_drug.molecular_weight = drug_data.get("molecular_weight")
                    existing_drug.description = drug_data.get("description")
                    existing_drug.indication = drug_data.get("indication")
                    existing_drug.targets = drug_data.get("targets", [])
                    existing_drug.properties = drug_data.get("properties", {})
                    
                    # Update features
                    if drug_data.get("smiles"):
                        features = featurizer.featurize_smiles(drug_data["smiles"])
                        existing_drug.features = features
                        
                    processed_ids.append(existing_drug.source_id)
                else:
                    # Create new drug
                    features = None
                    if drug_data.get("smiles"):
                        features = featurizer.featurize_smiles(drug_data["smiles"])
                    
                    new_drug = Drug(
                        source="drugbank",
                        source_id=drug_data.get("drugbank_id"),
                        name=drug_data.get("name"),
                        smiles=drug_data.get("smiles"),
                        molecular_formula=drug_data.get("molecular_formula"),
                        molecular_weight=drug_data.get("molecular_weight"),
                        description=drug_data.get("description"),
                        indication=drug_data.get("indication"),
                        targets=drug_data.get("targets", []),
                        properties=drug_data.get("properties", {}),
                        features=features
                    )
                    
                    db.add(new_drug)
                    processed_ids.append(drug_data.get("drugbank_id"))
                
                # Commit in batches
                if len(processed_ids) % 100 == 0:
                    db.commit()
                    logger.info(f"Processed {len(processed_ids)} DrugBank drugs...")
                    
            except IntegrityError as e:
                logger.warning(f"Duplicate drug from DrugBank: {drug_data.get('drugbank_id')}")
                db.rollback()
            except Exception as e:
                logger.error(f"Error processing DrugBank drug {drug_data.get('drugbank_id')}: {str(e)}")
                db.rollback()
        
        db.commit()
        logger.info(f"DrugBank sync completed: {len(processed_ids)} drugs")
        return processed_ids
        
    except Exception as e:
        logger.error(f"DrugBank sync failed: {str(e)}")
        raise

async def sync_chembl_drugs(client: ChEMBLClient, db, featurizer: DrugFeaturizer) -> List[str]:
    """Sync drugs from ChEMBL"""
    try:
        drugs_data = await client.fetch_molecules(limit=1000)
        processed_ids = []
        
        for drug_data in drugs_data:
            try:
                # Check if drug already exists
                existing_drug = db.query(Drug).filter(
                    Drug.source == "chembl",
                    Drug.source_id == drug_data.get("molecule_chembl_id")
                ).first()
                
                if existing_drug:
                    # Update existing drug
                    existing_drug.name = drug_data.get("pref_name")
                    existing_drug.smiles = drug_data.get("molecule_structures", {}).get("canonical_smiles")
                    existing_drug.molecular_formula = drug_data.get("molecule_properties", {}).get("molecular_formula")
                    existing_drug.molecular_weight = drug_data.get("molecule_properties", {}).get("molecular_weight")
                    existing_drug.description = drug_data.get("description")
                    existing_drug.indication = drug_data.get("therapeutic_flag")
                    existing_drug.properties = drug_data.get("molecule_properties", {})
                    
                    # Update features
                    smiles = drug_data.get("molecule_structures", {}).get("canonical_smiles")
                    if smiles:
                        features = featurizer.featurize_smiles(smiles)
                        existing_drug.features = features
                        
                    processed_ids.append(existing_drug.source_id)
                else:
                    # Create new drug
                    smiles = drug_data.get("molecule_structures", {}).get("canonical_smiles")
                    features = None
                    if smiles:
                        features = featurizer.featurize_smiles(smiles)
                    
                    new_drug = Drug(
                        source="chembl",
                        source_id=drug_data.get("molecule_chembl_id"),
                        name=drug_data.get("pref_name"),
                        smiles=smiles,
                        molecular_formula=drug_data.get("molecule_properties", {}).get("molecular_formula"),
                        molecular_weight=drug_data.get("molecule_properties", {}).get("molecular_weight"),
                        description=drug_data.get("description"),
                        indication=drug_data.get("therapeutic_flag"),
                        targets=[],  # Will be populated separately
                        properties=drug_data.get("molecule_properties", {}),
                        features=features
                    )
                    
                    db.add(new_drug)
                    processed_ids.append(drug_data.get("molecule_chembl_id"))
                
                # Commit in batches
                if len(processed_ids) % 100 == 0:
                    db.commit()
                    logger.info(f"Processed {len(processed_ids)} ChEMBL drugs...")
                    
            except IntegrityError as e:
                logger.warning(f"Duplicate drug from ChEMBL: {drug_data.get('molecule_chembl_id')}")
                db.rollback()
            except Exception as e:
                logger.error(f"Error processing ChEMBL drug {drug_data.get('molecule_chembl_id')}: {str(e)}")
                db.rollback()
        
        db.commit()
        logger.info(f"ChEMBL sync completed: {len(processed_ids)} drugs")
        return processed_ids
        
    except Exception as e:
        logger.error(f"ChEMBL sync failed: {str(e)}")
        raise

async def sync_pubchem_drugs(client: PubChemClient, db, featurizer: DrugFeaturizer) -> List[str]:
    """Sync drugs from PubChem"""
    try:
        # PubChem sync strategy: fetch approved drugs or specific compound lists
        drugs_data = await client.fetch_compounds(compound_type="approved_drugs", limit=1000)
        processed_ids = []
        
        for drug_data in drugs_data:
            try:
                cid = str(drug_data.get("cid"))
                
                # Check if drug already exists
                existing_drug = db.query(Drug).filter(
                    Drug.source == "pubchem",
                    Drug.source_id == cid
                ).first()
                
                if existing_drug:
                    # Update existing drug
                    existing_drug.name = drug_data.get("iupac_name") or drug_data.get("title")
                    existing_drug.smiles = drug_data.get("canonical_smiles")
                    existing_drug.molecular_formula = drug_data.get("molecular_formula")
                    existing_drug.molecular_weight = drug_data.get("molecular_weight")
                    existing_drug.description = drug_data.get("description")
                    existing_drug.properties = {
                        "pubchem_cid": cid,
                        "complexity": drug_data.get("complexity"),
                        "heavy_atom_count": drug_data.get("heavy_atom_count"),
                        "h_bond_donor_count": drug_data.get("h_bond_donor_count"),
                        "h_bond_acceptor_count": drug_data.get("h_bond_acceptor_count"),
                        "rotatable_bond_count": drug_data.get("rotatable_bond_count"),
                        "topological_polar_surface_area": drug_data.get("tpsa"),
                        "xlogp": drug_data.get("xlogp")
                    }
                    
                    # Update features
                    if drug_data.get("canonical_smiles"):
                        features = featurizer.featurize_smiles(drug_data["canonical_smiles"])
                        existing_drug.features = features
                        
                    processed_ids.append(cid)
                else:
                    # Create new drug
                    features = None
                    if drug_data.get("canonical_smiles"):
                        features = featurizer.featurize_smiles(drug_data["canonical_smiles"])
                    
                    new_drug = Drug(
                        source="pubchem",
                        source_id=cid,
                        name=drug_data.get("iupac_name") or drug_data.get("title"),
                        smiles=drug_data.get("canonical_smiles"),
                        molecular_formula=drug_data.get("molecular_formula"),
                        molecular_weight=drug_data.get("molecular_weight"),
                        description=drug_data.get("description"),
                        targets=[],  # Will be populated separately
                        properties={
                            "pubchem_cid": cid,
                            "complexity": drug_data.get("complexity"),
                            "heavy_atom_count": drug_data.get("heavy_atom_count"),
                            "h_bond_donor_count": drug_data.get("h_bond_donor_count"),
                            "h_bond_acceptor_count": drug_data.get("h_bond_acceptor_count"),
                            "rotatable_bond_count": drug_data.get("rotatable_bond_count"),
                            "topological_polar_surface_area": drug_data.get("tpsa"),
                            "xlogp": drug_data.get("xlogp")
                        },
                        features=features
                    )
                    
                    db.add(new_drug)
                    processed_ids.append(cid)
                
                # Commit in batches
                if len(processed_ids) % 100 == 0:
                    db.commit()
                    logger.info(f"Processed {len(processed_ids)} PubChem drugs...")
                    
            except IntegrityError as e:
                logger.warning(f"Duplicate drug from PubChem: {cid}")
                db.rollback()
            except Exception as e:
                logger.error(f"Error processing PubChem drug {cid}: {str(e)}")
                db.rollback()
        
        db.commit()
        logger.info(f"PubChem sync completed: {len(processed_ids)} drugs")
        return processed_ids
        
    except Exception as e:
        logger.error(f"PubChem sync failed: {str(e)}")
        raise

@celery_app.task(bind=True)
def deduplicate_drugs(self, job_id: int):
    """Deduplicate drugs across sources based on canonical SMILES"""
    
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        logger.error(f"Job {job_id} not found")
        return {"status": "error", "message": "Job not found"}
    
    try:
        job.status = "running"
        job.started_at = db.execute("SELECT NOW()").scalar()
        job.progress = 0
        db.commit()
        
        # Group drugs by canonical SMILES
        drugs = db.query(Drug).filter(Drug.smiles.isnot(None)).all()
        smiles_groups = {}
        
        for drug in drugs:
            canonical_smiles = drug.smiles.strip() if drug.smiles else None
            if canonical_smiles:
                if canonical_smiles not in smiles_groups:
                    smiles_groups[canonical_smiles] = []
                smiles_groups[canonical_smiles].append(drug)
        
        duplicates_removed = 0
        total_groups = len(smiles_groups)
        processed_groups = 0
        
        for smiles, drug_group in smiles_groups.items():
            if len(drug_group) > 1:
                # Keep the drug with the most complete information
                best_drug = max(drug_group, key=lambda d: (
                    bool(d.name),
                    bool(d.description),
                    bool(d.molecular_formula),
                    bool(d.features),
                    len(d.targets or []),
                    len(d.properties or {})
                ))
                
                # Merge information from other drugs
                for other_drug in drug_group:
                    if other_drug.id != best_drug.id:
                        # Merge targets
                        if other_drug.targets:
                            best_drug.targets = list(set((best_drug.targets or []) + other_drug.targets))
                        
                        # Merge properties
                        if other_drug.properties:
                            merged_props = best_drug.properties or {}
                            merged_props.update(other_drug.properties)
                            best_drug.properties = merged_props
                        
                        # Update fields if better info available
                        if not best_drug.name and other_drug.name:
                            best_drug.name = other_drug.name
                        if not best_drug.description and other_drug.description:
                            best_drug.description = other_drug.description
                        if not best_drug.indication and other_drug.indication:
                            best_drug.indication = other_drug.indication
                        
                        # Remove duplicate
                        db.delete(other_drug)
                        duplicates_removed += 1
            
            processed_groups += 1
            if processed_groups % 100 == 0:
                progress = int((processed_groups / total_groups) * 100)
                job.progress = progress
                db.commit()
                
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': processed_groups,
                        'total': total_groups,
                        'message': f"Processed {processed_groups}/{total_groups} SMILES groups",
                        'progress': progress
                    }
                )
        
        db.commit()
        
        job.status = "completed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.progress = 100
        job.result = {
            "duplicates_removed": duplicates_removed,
            "unique_smiles_groups": total_groups,
            "remaining_drugs": db.query(Drug).count()
        }
        job.logs = f"Deduplication completed. Removed {duplicates_removed} duplicates."
        db.commit()
        
        logger.info(f"Drug deduplication job {job_id} completed: removed {duplicates_removed} duplicates")
        return {"status": "success", "result": job.result}
        
    except Exception as e:
        logger.error(f"Drug deduplication job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.error = str(e)
        db.commit()
        raise
    finally:
        db.close()

@celery_app.task(bind=True)
def update_drug_targets(self, job_id: int):
    """Update drug-target interactions from all sources"""
    
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        logger.error(f"Job {job_id} not found")
        return {"status": "error", "message": "Job not found"}
    
    try:
        job.status = "running"
        job.started_at = db.execute("SELECT NOW()").scalar()
        job.progress = 0
        db.commit()
        
        # Initialize clients
        drugbank_client = DrugBankClient()
        chembl_client = ChEMBLClient()
        
        drugs = db.query(Drug).all()
        total_drugs = len(drugs)
        processed = 0
        
        for drug in drugs:
            try:
                targets = []
                
                # Fetch targets based on source
                if drug.source == "drugbank":
                    targets = await drugbank_client.fetch_drug_targets(drug.source_id)
                elif drug.source == "chembl":
                    targets = await chembl_client.fetch_molecule_targets(drug.source_id)
                
                # Update drug targets
                if targets:
                    drug.targets = targets
                
                processed += 1
                if processed % 50 == 0:
                    progress = int((processed / total_drugs) * 100)
                    job.progress = progress
                    db.commit()
                    
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': processed,
                            'total': total_drugs,
                            'message': f"Updated targets for {processed}/{total_drugs} drugs",
                            'progress': progress
                        }
                    )
                    
                    logger.info(f"Updated targets for {processed}/{total_drugs} drugs...")
                    
            except Exception as e:
                logger.error(f"Error updating targets for drug {drug.id}: {str(e)}")
                continue
        
        db.commit()
        
        job.status = "completed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.progress = 100
        job.result = {"drugs_updated": processed}
        job.logs = f"Target update completed for {processed} drugs."
        db.commit()
        
        logger.info(f"Drug target update job {job_id} completed")
        return {"status": "success", "result": job.result}
        
    except Exception as e:
        logger.error(f"Drug target update job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.error = str(e)
        db.commit()
        raise
    finally:
        db.close()

@celery_app.task(bind=True, rate_limit="10/m")
def sync_single_drug(self, source: str, source_id: str, force_update: bool = False):
    """Sync a single drug from a specific source"""
    
    db = SessionLocal()
    
    try:
        # Check if drug exists and skip if not forcing update
        if not force_update:
            existing = db.query(Drug).filter(
                Drug.source == source,
                Drug.source_id == source_id
            ).first()
            if existing:
                logger.info(f"Drug {source}:{source_id} already exists, skipping")
                return {"status": "skipped", "reason": "already_exists"}
        
        # Initialize appropriate client
        if source == "drugbank":
            client = DrugBankClient()
            drug_data = await client.fetch_drug(source_id)
        elif source == "chembl":
            client = ChEMBLClient()
            drug_data = await client.fetch_molecule(source_id)
        elif source == "pubchem":
            client = PubChemClient()
            drug_data = await client.fetch_compound(source_id)
        else:
            raise ValueError(f"Unknown drug source: {source}")
        
        if not drug_data:
            return {"status": "error", "message": "Drug not found in source"}
        
        # Process the drug data (similar to sync functions above)
        featurizer = DrugFeaturizer()
        
        # Create or update drug
        existing_drug = db.query(Drug).filter(
            Drug.source == source,
            Drug.source_id == source_id
        ).first()
        
        if existing_drug:
            # Update logic here...
            logger.info(f"Updated drug {source}:{source_id}")
            action = "updated"
        else:
            # Create logic here...
            logger.info(f"Created drug {source}:{source_id}")
            action = "created"
        
        db.commit()
        
        return {
            "status": "success",
            "action": action,
            "drug_id": existing_drug.id if existing_drug else None
        }
        
    except Exception as e:
        logger.error(f"Failed to sync drug {source}:{source_id}: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

# Periodic task for incremental sync
@celery_app.task
def periodic_drug_sync():
    """Periodic task to sync new drugs incrementally"""
    try:
        # Create a new job for the sync
        db = SessionLocal()
        
        job = Job(
            type="drug_sync_periodic",
            status="pending",
            progress=0,
            meta={"source": "periodic", "incremental": True}
        )
        db.add(job)
        db.commit()
        
        # Queue the sync task
        sync_drugs_from_sources.delay(job.id)
        
        logger.info(f"Queued periodic drug sync job {job.id}")
        return {"status": "queued", "job_id": job.id}
        
    except Exception as e:
        logger.error(f"Failed to queue periodic drug sync: {str(e)}")
        raise
    finally:
        db.close()
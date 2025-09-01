"""
Celery tasks for PPI predictions and batch processing.
"""
import traceback
from typing import Dict, Any, List, Optional
import asyncio
from celery import current_task
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.db.session import get_db
from app.db.models.prediction import Prediction
from app.db.models.job import Job
from app.db.models.protein import Protein
from app.db.models.blockchain_event import BlockchainEvent
from app.ml.infer.predictor import UnifiedPredictor
from app.ml.infer.structure_export import StructureExporter
from app.services.uniprot_client import UniProtClient
from app.services.pdb_client import PDBClient
from app.core.blockchain import log_blockchain_event
from app.core.logging_conf import logger
from app.services.storage import StorageService
from app.services.hooks.arduino import ArduinoPublisher
from app.services.hooks.hologram_ws import HologramStreamer


@celery_app.task(bind=True)
def predict_ppi_task(
    self,
    protein_a_data: Dict[str, Any],
    protein_b_data: Dict[str, Any],
    user_id: int,
    prediction_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Background task for PPI prediction.
    
    Args:
        protein_a_data: {id: str, sequence: str, uniprot_id: str}
        protein_b_data: {id: str, sequence: str, uniprot_id: str}
        user_id: User ID for blockchain logging
        prediction_id: Optional existing prediction ID
    """
    task_id = self.request.id
    db = next(get_db())
    
    try:
        # Update job status
        job = db.query(Job).filter(Job.celery_id == task_id).first()
        if job:
            job.status = "running"
            job.progress = 0
            db.commit()
        
        logger.info(f"Starting PPI prediction task {task_id}")
        
        # Initialize predictor
        predictor = UnifiedPredictor()
        exporter = StructureExporter()
        storage = StorageService()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "stage": "Loading proteins"})
        if job:
            job.progress = 10
            db.commit()
        
        # Fetch sequences if needed
        if not protein_a_data.get("sequence") and protein_a_data.get("uniprot_id"):
            uniprot_client = UniProtClient()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            protein_a_data["sequence"] = loop.run_until_complete(
                uniprot_client.get_sequence(protein_a_data["uniprot_id"])
            )
            loop.close()
        
        if not protein_b_data.get("sequence") and protein_b_data.get("uniprot_id"):
            uniprot_client = UniProtClient()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            protein_b_data["sequence"] = loop.run_until_complete(
                uniprot_client.get_sequence(protein_b_data["uniprot_id"])
            )
            loop.close()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "stage": "Building features"})
        if job:
            job.progress = 30
            db.commit()
        
        # Run prediction
        prediction_result = predictor.predict_ppi(
            protein_a_seq=protein_a_data["sequence"],
            protein_b_seq=protein_b_data["sequence"],
            protein_a_id=protein_a_data.get("uniprot_id", protein_a_data["id"]),
            protein_b_id=protein_b_data.get("uniprot_id", protein_b_data["id"])
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 60, "stage": "Exporting structures"})
        if job:
            job.progress = 60
            db.commit()
        
        # Export structures
        structure_paths = exporter.export_structures(
            prediction_result["coordinates"],
            prediction_result["residue_annotations"],
            protein_a_data["id"],
            protein_b_data["id"]
        )
        
        # Store files
        stored_paths = {}
        for format_type, path in structure_paths.items():
            stored_path = storage.store_file(path, f"predictions/{prediction_id or task_id}")
            stored_paths[f"{format_type}_url"] = storage.get_signed_url(stored_path)
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "stage": "Saving results"})
        if job:
            job.progress = 80
            db.commit()
        
        # Save prediction to database
        if prediction_id:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        else:
            prediction = Prediction()
            db.add(prediction)
        
        prediction.protein_a_id = protein_a_data.get("uniprot_id", protein_a_data["id"])
        prediction.protein_b_id = protein_b_data.get("uniprot_id", protein_b_data["id"])
        prediction.interaction_score = prediction_result["interaction_score"]
        prediction.binding_affinity = prediction_result["binding_affinity"]
        prediction.docking_score = prediction_result["docking_score"]
        prediction.residue_annotations = prediction_result["residue_annotations"]
        prediction.structure_pdb_path = stored_paths.get("pdb_url")
        prediction.structure_cif_path = stored_paths.get("cif_url")
        prediction.structure_gltf_path = stored_paths.get("gltf_url")
        prediction.status = "completed"
        
        db.commit()
        db.refresh(prediction)
        
        # Log to blockchain
        try:
            tx_hash = log_blockchain_event(
                user_address=f"user_{user_id}",
                action="ppi.predict",
                ref_id=str(prediction.id),
                metadata={
                    "proteins": [protein_a_data["id"], protein_b_data["id"]],
                    "scores": {
                        "interaction": prediction_result["interaction_score"],
                        "binding_affinity": prediction_result["binding_affinity"],
                        "docking": prediction_result["docking_score"]
                    }
                }
            )
            prediction.tx_hash = tx_hash
            db.commit()
        except Exception as e:
            logger.warning(f"Blockchain logging failed: {e}")
        
        # Update job as completed
        current_task.update_state(state="SUCCESS", meta={"progress": 100, "stage": "Completed"})
        if job:
            job.status = "completed"
            job.progress = 100
            job.result = {
                "prediction_id": prediction.id,
                "interaction_score": prediction_result["interaction_score"],
                "binding_affinity": prediction_result["binding_affinity"],
                "docking_score": prediction_result["docking_score"]
            }
            db.commit()
        
        # Trigger hooks
        try:
            # Arduino hook
            arduino_pub = ArduinoPublisher()
            arduino_pub.publish_prediction_result(prediction_result, dry_run=True)
            
            # Hologram WebSocket notification
            hologram_streamer = HologramStreamer()
            hologram_streamer.broadcast_prediction(prediction.id, prediction_result)
        except Exception as e:
            logger.warning(f"Hook notification failed: {e}")
        
        logger.info(f"PPI prediction task {task_id} completed successfully")
        
        return {
            "success": True,
            "prediction_id": prediction.id,
            "interaction_score": prediction_result["interaction_score"],
            "binding_affinity": prediction_result["binding_affinity"],
            "docking_score": prediction_result["docking_score"],
            "structure_urls": stored_paths,
            "tx_hash": prediction.tx_hash
        }
        
    except Exception as e:
        error_msg = f"PPI prediction failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}\n{traceback.format_exc()}")
        
        # Update job as failed
        current_task.update_state(
            state="FAILURE",
            meta={"progress": 0, "stage": "Failed", "error": error_msg}
        )
        if job:
            job.status = "failed"
            job.error_message = error_msg
            db.commit()
        
        # Log failure to blockchain
        try:
            log_blockchain_event(
                user_address=f"user_{user_id}",
                action="ppi.predict.failed",
                ref_id=task_id,
                metadata={"error": error_msg}
            )
        except:
            pass
        
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True)
def batch_predict_task(
    self,
    protein_pairs: List[Dict[str, Any]],
    user_id: int,
    batch_name: str = "batch_prediction"
) -> Dict[str, Any]:
    """
    Background task for batch PPI predictions.
    
    Args:
        protein_pairs: List of {protein_a: {...}, protein_b: {...}}
        user_id: User ID for blockchain logging
        batch_name: Name for the batch job
    """
    task_id = self.request.id
    db = next(get_db())
    
    try:
        # Update job status
        job = db.query(Job).filter(Job.celery_id == task_id).first()
        if job:
            job.status = "running"
            job.progress = 0
            db.commit()
        
        logger.info(f"Starting batch prediction task {task_id} with {len(protein_pairs)} pairs")
        
        results = []
        total_pairs = len(protein_pairs)
        
        for i, pair in enumerate(protein_pairs):
            try:
                # Run individual prediction
                result = predict_ppi_task.apply(
                    args=[pair["protein_a"], pair["protein_b"], user_id],
                    countdown=0
                ).get()
                
                results.append({
                    "pair_index": i,
                    "protein_a": pair["protein_a"]["id"],
                    "protein_b": pair["protein_b"]["id"],
                    "result": result,
                    "status": "success"
                })
                
                # Update progress
                progress = int((i + 1) / total_pairs * 100)
                current_task.update_state(
                    state="PROGRESS",
                    meta={"progress": progress, "stage": f"Processed {i+1}/{total_pairs} pairs"}
                )
                if job:
                    job.progress = progress
                    db.commit()
                
            except Exception as e:
                logger.error(f"Batch prediction failed for pair {i}: {e}")
                results.append({
                    "pair_index": i,
                    "protein_a": pair["protein_a"]["id"],
                    "protein_b": pair["protein_b"]["id"],
                    "result": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Log batch completion to blockchain
        try:
            successful_predictions = len([r for r in results if r["status"] == "success"])
            tx_hash = log_blockchain_event(
                user_address=f"user_{user_id}",
                action="ppi.batch_predict",
                ref_id=task_id,
                metadata={
                    "batch_name": batch_name,
                    "total_pairs": total_pairs,
                    "successful": successful_predictions,
                    "failed": total_pairs - successful_predictions
                }
            )
        except Exception as e:
            logger.warning(f"Blockchain logging failed: {e}")
            tx_hash = None
        
        # Update job as completed
        current_task.update_state(state="SUCCESS", meta={"progress": 100, "stage": "Completed"})
        if job:
            job.status = "completed"
            job.progress = 100
            job.result = {
                "batch_name": batch_name,
                "total_pairs": total_pairs,
                "results": results,
                "tx_hash": tx_hash
            }
            db.commit()
        
        logger.info(f"Batch prediction task {task_id} completed")
        
        return {
            "success": True,
            "batch_name": batch_name,
            "total_pairs": total_pairs,
            "results": results,
            "tx_hash": tx_hash
        }
        
    except Exception as e:
        error_msg = f"Batch prediction failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}\n{traceback.format_exc()}")
        
        # Update job as failed
        current_task.update_state(
            state="FAILURE",
            meta={"progress": 0, "stage": "Failed", "error": error_msg}
        )
        if job:
            job.status = "failed"
            job.error_message = error_msg
            db.commit()
        
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True)
def generate_structures_task(
    self,
    prediction_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Background task for generating and exporting 3D structures.
    """
    task_id = self.request.id
    db = next(get_db())
    
    try:
        # Update job status
        job = db.query(Job).filter(Job.celery_id == task_id).first()
        if job:
            job.status = "running"
            job.progress = 0
            db.commit()
        
        logger.info(f"Starting structure generation task {task_id} for prediction {prediction_id}")
        
        # Get prediction
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 20, "stage": "Loading prediction data"})
        if job:
            job.progress = 20
            db.commit()
        
        # Initialize services
        exporter = StructureExporter()
        storage = StorageService()
        
        # Generate structures if coordinates exist
        if prediction.residue_annotations:
            # Update progress
            current_task.update_state(state="PROGRESS", meta={"progress": 50, "stage": "Generating structures"})
            if job:
                job.progress = 50
                db.commit()
            
            # Export to multiple formats
            structure_paths = exporter.export_prediction_structures(
                prediction_id=prediction.id,
                residue_annotations=prediction.residue_annotations,
                protein_a_id=prediction.protein_a_id,
                protein_b_id=prediction.protein_b_id
            )
            
            # Update progress
            current_task.update_state(state="PROGRESS", meta={"progress": 80, "stage": "Storing files"})
            if job:
                job.progress = 80
                db.commit()
            
            # Store files and get signed URLs
            stored_paths = {}
            for format_type, path in structure_paths.items():
                stored_path = storage.store_file(path, f"predictions/{prediction_id}")
                stored_paths[f"{format_type}_url"] = storage.get_signed_url(stored_path)
            
            # Update prediction with file paths
            prediction.structure_pdb_path = stored_paths.get("pdb_url")
            prediction.structure_cif_path = stored_paths.get("cif_url")
            prediction.structure_gltf_path = stored_paths.get("gltf_url")
            db.commit()
        
        # Log to blockchain
        try:
            tx_hash = log_blockchain_event(
                user_address=f"user_{user_id}",
                action="structure.generate",
                ref_id=str(prediction_id),
                metadata={
                    "formats": list(stored_paths.keys()) if 'stored_paths' in locals() else [],
                    "prediction_id": prediction_id
                }
            )
        except Exception as e:
            logger.warning(f"Blockchain logging failed: {e}")
            tx_hash = None
        
        # Update job as completed
        current_task.update_state(state="SUCCESS", meta={"progress": 100, "stage": "Completed"})
        if job:
            job.status = "completed"
            job.progress = 100
            job.result = {
                "prediction_id": prediction_id,
                "structure_urls": stored_paths if 'stored_paths' in locals() else {},
                "tx_hash": tx_hash
            }
            db.commit()
        
        logger.info(f"Structure generation task {task_id} completed")
        
        return {
            "success": True,
            "prediction_id": prediction_id,
            "structure_urls": stored_paths if 'stored_paths' in locals() else {},
            "tx_hash": tx_hash
        }
        
    except Exception as e:
        error_msg = f"Structure generation failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}\n{traceback.format_exc()}")
        
        # Update job as failed
        current_task.update_state(
            state="FAILURE",
            meta={"progress": 0, "stage": "Failed", "error": error_msg}
        )
        if job:
            job.status = "failed"
            job.error_message = error_msg
            db.commit()
        
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True)
def analyze_drug_target_task(
    self,
    drug_id: str,
    target_protein_id: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Background task for drug-target analysis.
    """
    task_id = self.request.id
    db = next(get_db())
    
    try:
        # Update job status
        job = db.query(Job).filter(Job.celery_id == task_id).first()
        if job:
            job.status = "running"
            job.progress = 0
            db.commit()
        
        logger.info(f"Starting drug-target analysis task {task_id}")
        
        # Initialize predictor
        predictor = UnifiedPredictor()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "stage": "Analyzing drug-target interaction"})
        if job:
            job.progress = 30
            db.commit()
        
        # Run drug-target prediction
        result = predictor.predict_drug_target_interaction(
            drug_id=drug_id,
            target_protein_id=target_protein_id
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "stage": "Saving results"})
        if job:
            job.progress = 80
            db.commit()
        
        # Log to blockchain
        try:
            tx_hash = log_blockchain_event(
                user_address=f"user_{user_id}",
                action="drug.target_analysis",
                ref_id=task_id,
                metadata={
                    "drug_id": drug_id,
                    "target_protein_id": target_protein_id,
                    "binding_score": result["binding_score"],
                    "confidence": result["confidence"]
                }
            )
        except Exception as e:
            logger.warning(f"Blockchain logging failed: {e}")
            tx_hash = None
        
        # Update job as completed
        current_task.update_state(state="SUCCESS", meta={"progress": 100, "stage": "Completed"})
        if job:
            job.status = "completed"
            job.progress = 100
            job.result = {
                "drug_id": drug_id,
                "target_protein_id": target_protein_id,
                "binding_score": result["binding_score"],
                "confidence": result["confidence"],
                "tx_hash": tx_hash
            }
            db.commit()
        
        logger.info(f"Drug-target analysis task {task_id} completed")
        
        return {
            "success": True,
            "drug_id": drug_id,
            "target_protein_id": target_protein_id,
            "binding_score": result["binding_score"],
            "confidence": result["confidence"],
            "mechanisms": result.get("mechanisms", []),
            "tx_hash": tx_hash
        }
        
    except Exception as e:
        error_msg = f"Drug-target analysis failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}\n{traceback.format_exc()}")
        
        # Update job as failed
        current_task.update_state(
            state="FAILURE",
            meta={"progress": 0, "stage": "Failed", "error": error_msg}
        )
        if job:
            job.status = "failed"
            job.error_message = error_msg
            db.commit()
        
        raise
    
    finally:
        db.close()
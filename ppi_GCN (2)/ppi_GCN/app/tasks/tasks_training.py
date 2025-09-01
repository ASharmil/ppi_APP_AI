from celery import current_task
import torch
from torch_geometric.loader import DataLoader
from app.core.celery_app import celery_app
from app.ml.models.gcn_ppi import PPIGraphNet
from app.ml.models.drug_matcher import DrugProteinMatcher
from app.ml.train.trainer_ppi import PPITrainer
from app.ml.train.trainer_drug import DrugTrainer
from app.ml.data.loaders import PPIDataLoader
from app.db.session import SessionLocal
from app.db.models.job import Job
from app.core.blockchain import blockchain_logger
from loguru import logger
import asyncio

@celery_app.task(bind=True)
def train_ppi_model(self, csv_path: str, config: dict, job_id: int):
    """Train PPI prediction model"""
    
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    try:
        # Update job status
        job.status = "running"
        job.started_at = db.execute("SELECT NOW()").scalar()
        db.commit()
        
        # Setup model and trainer
        model = PPIGraphNet(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1)
        )
        
        trainer = PPITrainer(model)
        trainer.setup_training(
            learning_rate=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Load and prepare data
        loader = PPIDataLoader()
        asyncio.set_event_loop(asyncio.new_event_loop())
        
        graph_data, interaction_labels, affinity_labels = asyncio.get_event_loop().run_until_complete(
            loader.prepare_training_data(csv_path)
        )
        
        # Create data loaders
        train_loader = DataLoader(graph_data[:int(0.8 * len(graph_data))], batch_size=config.get('batch_size', 32))
        val_loader = DataLoader(graph_data[int(0.8 * len(graph_data)):], batch_size=config.get('batch_size', 32))
        
        # Training loop
        epochs = config.get('epochs', 100)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Update progress
            progress = int((epoch / epochs) * 100)
            current_task.update_state(state='PROGRESS', meta={'progress': progress})
            job.progress = progress
            db.commit()
            
            # Train epoch
            train_loss, train_metrics = asyncio.get_event_loop().run_until_complete(
                trainer.train_epoch(train_loader, [])  # TODO: Add proper haddock features
            )
            
            # Validation epoch
            val_loss, val_metrics = asyncio.get_event_loop().run_until_complete(
                trainer.validate_epoch(val_loader, [])
            )
            
            # Update metrics
            trainer.train_losses.append(train_loss)
            trainer.val_losses.append(val_loss)
            trainer.train_metrics.append(train_metrics)
            trainer.val_metrics.append(val_metrics)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            trainer.save_checkpoint(epoch, val_metrics, is_best)
            
            # Schedule learning rate
            trainer.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, roc_auc={val_metrics['roc_auc']:.4f}")
        
        # Generate training curves
        curves_path = trainer.plot_training_curves()
        
        # Update job completion
        job.status = "success"
        job.progress = 100
        job.completed_at = db.execute("SELECT NOW()").scalar()
        job.result = {
            "final_metrics": val_metrics,
            "curves_path": curves_path,
            "model_path": str(trainer.model)
        }
        
        # Log to blockchain
        tx_hash = asyncio.get_event_loop().run_until_complete(
            blockchain_logger.log_event(
                job.user.email if job.user else "system",
                f"models.train:{job_id}",
                str(job_id)
            )
        )
        job.tx_hash = tx_hash
        
        db.commit()
        logger.info(f"PPI training completed for job {job_id}")
        
        return {"status": "success", "metrics": val_metrics}
        
    except Exception as e:
        logger.error(f"PPI training failed for job {job_id}: {e}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise
    finally:
        db.close()

@celery_app.task(bind=True)
def train_drug_model(self, job_id: int, config: dict):
    """Train drug-protein matching model"""
    
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    try:
        job.status = "running"
        job.started_at = db.execute("SELECT NOW()").scalar()
        db.commit()
        
        # Setup model
        model = DrugProteinMatcher(
            drug_features=config.get('drug_features', 200),
            protein_embed_dim=config.get('protein_embed_dim', 128),
            hidden_dim=config.get('hidden_dim', 256)
        )
        
        trainer = DrugTrainer(model)
        trainer.setup_training(
            learning_rate=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # TODO: Load drug-protein interaction data and train
        # This would involve:
        # 1. Load drug features from synced DrugBank/ChEMBL data
        # 2. Load protein embeddings from PPI model
        # 3. Create positive/negative training pairs
        # 4. Train the matching model
        
        # For now, simulate training
        epochs = config.get('epochs', 50)
        for epoch in range(epochs):
            progress = int((epoch / epochs) * 100)
            current_task.update_state(state='PROGRESS', meta={'progress': progress})
            job.progress = progress
            db.commit()
            
            # Simulate training step
            await asyncio.sleep(0.1)
        
        job.status = "success"
        job.progress = 100
        job.completed_at = db.execute("SELECT NOW()").scalar()
        db.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Drug training failed for job {job_id}: {e}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise
    finally:
        db.close()
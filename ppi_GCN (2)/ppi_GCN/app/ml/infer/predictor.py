import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from loguru import logger
from app.ml.models.gcn_ppi import PPIGraphNet, PPIDataBuilder
from app.ml.models.drug_matcher import DrugProteinMatcher, DrugFeaturizer
from app.ml.data.featurizers import HADDOCKFeatureBuilder
from app.core.config import settings

class PPIPredictor:
    def __init__(self):
        self.ppi_model = None
        self.drug_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_builder = PPIDataBuilder()
        self.haddock_builder = HADDOCKFeatureBuilder()
        self.drug_featurizer = DrugFeaturizer()
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load PPI model
            ppi_path = Path(settings.MODELS_PATH) / "ppi_model_best.pt"
            if ppi_path.exists():
                checkpoint = torch.load(ppi_path, map_location=self.device)
                self.ppi_model = PPIGraphNet()
                self.ppi_model.load_state_dict(checkpoint['model_state_dict'])
                self.ppi_model.to(self.device)
                self.ppi_model.eval()
                logger.info("PPI model loaded successfully")
            
            # Load drug model
            drug_path = Path(settings.MODELS_PATH) / "drug_matcher_best.pt"
            if drug_path.exists():
                checkpoint = torch.load(drug_path, map_location=self.device)
                self.drug_model = DrugProteinMatcher()
                self.drug_model.load_state_dict(checkpoint['model_state_dict'])
                self.drug_model.to(self.device)
                self.drug_model.eval()
                logger.info("Drug model loaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    async def predict_ppi(
        self,
        sequence_a: str,
        sequence_b: str,
        haddock_features: Optional[Dict] = None
    ) -> Dict:
        """Predict protein-protein interaction"""
        
        if self.ppi_model is None:
            raise ValueError("PPI model not loaded")
        
        try:
            # Build graph data
            graph_data, haddock_tensor = self.data_builder.build_graph_data(
                sequence_a, sequence_b, haddock_features
            )
            
            # Move to device
            graph_data = graph_data.to(self.device)
            haddock_tensor = haddock_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.ppi_model(graph_data, haddock_tensor)
            
            # Extract residue-level annotations
            residue_annotations = self._extract_residue_annotations(
                outputs, len(sequence_a), len(sequence_b)
            )
            
            return {
                'interaction_score': float(outputs['interaction_score'].cpu().numpy()),
                'binding_affinity': float(outputs['binding_affinity'].cpu().numpy()),
                'confidence': float(torch.sigmoid(outputs['interaction_score']).cpu().numpy()),
                'residue_annotations': residue_annotations,
                'haddock_features': haddock_features or self.haddock_builder.compute_features(sequence_a, sequence_b)
            }
            
        except Exception as e:
            logger.error(f"PPI prediction failed: {e}")
            raise
    
    def _extract_residue_annotations(self, outputs: Dict, len_a: int, len_b: int) -> Dict:
        """Extract per-residue interaction probabilities"""
        attention_weights = outputs['residue_attention'].cpu().numpy()
        node_embeddings = outputs['node_embeddings'].cpu().numpy()
        
        # Split by protein
        protein_a_weights = attention_weights[:len_a]
        protein_b_weights = attention_weights[len_a:]
        
        # Compute interaction probabilities per residue
        a_interactions = np.mean(protein_a_weights, axis=1)
        b_interactions = np.mean(protein_b_weights, axis=1)
        
        return {
            'protein_a': {
                'residue_scores': a_interactions.tolist(),
                'high_confidence_residues': np.where(a_interactions > 0.7)[0].tolist()
            },
            'protein_b': {
                'residue_scores': b_interactions.tolist(),
                'high_confidence_residues': np.where(b_interactions > 0.7)[0].tolist()
            }
        }
    
    async def predict_drug_affinity(
        self,
        drug_smiles: str,
        protein_embedding: np.ndarray
    ) -> Dict:
        """Predict drug-protein affinity"""
        
        if self.drug_model is None:
            raise ValueError("Drug model not loaded")
        
        try:
            # Compute drug features
            drug_features = self.drug_featurizer.compute_molecular_descriptors(drug_smiles)
            
            # Convert to tensors
            drug_tensor = torch.tensor(drug_features).unsqueeze(0).to(self.device)
            protein_tensor = torch.tensor(protein_embedding).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.drug_model(drug_tensor, protein_tensor)
            
            return {
                'affinity_score': float(outputs['affinity_score'].cpu().numpy()),
                'confidence': float(outputs['confidence'].cpu().numpy()),
                'attention_weights': outputs['attention_weights'].cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Drug affinity prediction failed: {e}")
            raise
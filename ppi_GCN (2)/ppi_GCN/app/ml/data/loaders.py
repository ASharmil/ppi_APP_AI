import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import asyncio
import httpx
from Bio import SeqIO
from io import StringIO
from app.services.uniprot_client import UniProtClient
from app.services.pdb_client import PDBClient
from app.ml.models.gcn_ppi import PPIDataBuilder
from loguru import logger

class PPIDataLoader:
    def __init__(self):
        self.uniprot_client = UniProtClient()
        self.pdb_client = PDBClient()
        self.data_builder = PPIDataBuilder()
    
    async def load_csv_data(self, csv_path: str) -> Tuple[List[Dict], List[float], List[float]]:
        """Load HADDOCK features and labels from CSV"""
        try:
            df = pd.read_csv(csv_path)
            
            # Expected CSV columns: protein_a_id, protein_b_id, + HADDOCK features + labels
            required_cols = ['protein_a_id', 'protein_b_id', 'interaction_label', 'binding_affinity']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            
            # Extract HADDOCK feature columns (numeric columns excluding IDs and labels)
            feature_cols = [col for col in df.columns 
                          if col not in ['protein_a_id', 'protein_b_id', 'interaction_label', 'binding_affinity']
                          and df[col].dtype in ['float64', 'int64']]
            
            logger.info(f"Found HADDOCK feature columns: {feature_cols}")
            
            data_samples = []
            interaction_labels = df['interaction_label'].tolist()
            binding_affinities = df['binding_affinity'].tolist()
            
            for _, row in df.iterrows():
                sample = {
                    'protein_a_id': row['protein_a_id'],
                    'protein_b_id': row['protein_b_id'],
                    'haddock_features': {col: row[col] for col in feature_cols}
                }
                data_samples.append(sample)
            
            return data_samples, interaction_labels, binding_affinities
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    async def fetch_protein_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """Fetch amino acid sequences for protein IDs"""
        sequences = {}
        
        for protein_id in protein_ids:
            try:
                # Try UniProt first
                sequence = await self.uniprot_client.get_sequence(protein_id)
                if sequence:
                    sequences[protein_id] = sequence
                    continue
                
                # Try PDB if UniProt fails
                sequence = await self.pdb_client.get_sequence(protein_id)
                if sequence:
                    sequences[protein_id] = sequence
                else:
                    logger.warning(f"Could not fetch sequence for {protein_id}")
                    
            except Exception as e:
                logger.error(f"Error fetching sequence for {protein_id}: {e}")
        
        return sequences
    
    async def prepare_training_data(self, csv_path: str) -> Tuple[List, List, List]:
        """Prepare complete training dataset with sequences and features"""
        
        # Load CSV data
        data_samples, interaction_labels, binding_affinities = await self.load_csv_data(csv_path)
        
        # Extract unique protein IDs
        protein_ids = set()
        for sample in data_samples:
            protein_ids.add(sample['protein_a_id'])
            protein_ids.add(sample['protein_b_id'])
        
        # Fetch sequences
        logger.info(f"Fetching sequences for {len(protein_ids)} unique proteins...")
        sequences = await self.fetch_protein_sequences(list(protein_ids))
        
        # Build graph data
        graph_data_list = []
        haddock_features_list = []
        valid_indices = []
        
        for i, sample in enumerate(data_samples):
            seq_a = sequences.get(sample['protein_a_id'])
            seq_b = sequences.get(sample['protein_b_id'])
            
            if seq_a and seq_b:
                try:
                    graph_data, haddock_tensor = self.data_builder.build_graph_data(
                        seq_a, seq_b, sample['haddock_features']
                    )
                    graph_data_list.append(graph_data)
                    haddock_features_list.append(haddock_tensor)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Error building graph for sample {i}: {e}")
        
        # Filter labels to match valid samples
        valid_interaction_labels = [interaction_labels[i] for i in valid_indices]
        valid_binding_affinities = [binding_affinities[i] for i in valid_indices]
        
        logger.info(f"Prepared {len(graph_data_list)} valid training samples")
        
        return graph_data_list, valid_interaction_labels, valid_binding_affinities
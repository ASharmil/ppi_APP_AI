import numpy as np
import torch
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from app.ml.models.gcn_ppi import PPIDataBuilder

class HADDOCKFeatureBuilder:
    """Build HADDOCK-style docking features from protein structures"""
    
    FEATURE_NAMES = [
        'Evdw',         # Van der Waals energy
        'Eelec',        # Electrostatic energy
        'Eair',         # Ambiguous interaction restraints
        'Edih',         # Dihedral angle energy
        'Ecoup',        # Coupling energy
        'Esani',        # Sanity check energy
        'Evdw_int',     # Intermolecular VdW
        'Eelec_int',    # Intermolecular electrostatic
        'Edesolvation', # Desolvation energy
        'BSA',          # Buried surface area
        'RMSD',         # RMSD from reference
        'FCC'           # Fraction of common contacts
    ]
    
    @classmethod
    def compute_features(cls, protein_a_seq: str, protein_b_seq: str) -> Dict[str, float]:
        """Compute HADDOCK-style features from sequences (simplified)"""
        
        # This is a simplified feature computation
        # In production, you'd use actual structural data and docking software
        
        features = {}
        
        # Simple heuristics based on sequence properties
        len_a, len_b = len(protein_a_seq), len(protein_b_seq)
        
        # Van der Waals (based on hydrophobic residues)
        hydrophobic = set('AILVMFYW')
        hydrophobic_a = sum(1 for aa in protein_a_seq if aa in hydrophobic) / len_a
        hydrophobic_b = sum(1 for aa in protein_b_seq if aa in hydrophobic) / len_b
        features['Evdw'] = -20.0 * hydrophobic_a * hydrophobic_b * np.sqrt(len_a * len_b)
        
        # Electrostatic (based on charged residues)
        positive = set('RK')
        negative = set('DE')
        charge_a = sum(1 for aa in protein_a_seq if aa in positive) - sum(1 for aa in protein_a_seq if aa in negative)
        charge_b = sum(1 for aa in protein_b_seq if aa in positive) - sum(1 for aa in protein_b_seq if aa in negative)
        features['Eelec'] = -10.0 * abs(charge_a * charge_b) / np.sqrt(len_a * len_b)
        
        # Other features (simplified estimates)
        features['Eair'] = np.random.normal(-5.0, 2.0)
        features['Edih'] = np.random.normal(10.0, 3.0)
        features['Ecoup'] = np.random.normal(0.0, 1.0)
        features['Esani'] = np.random.normal(0.0, 0.5)
        features['Evdw_int'] = features['Evdw'] * 0.7
        features['Eelec_int'] = features['Eelec'] * 0.8
        features['Edesolvation'] = np.random.normal(5.0, 2.0)
        features['BSA'] = min(len_a, len_b) * np.random.uniform(0.1, 0.3)
        features['RMSD'] = np.random.uniform(1.0, 5.0)
        features['FCC'] = np.random.uniform(0.3, 0.8)
        
        return features

class DrugFeaturizer:
    """Extract molecular features from drug SMILES"""
    
    @classmethod
    def compute_molecular_descriptors(cls, smiles: str) -> np.ndarray:
        """Compute comprehensive molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(200)
            
            # Core descriptors
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.BertzCT(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.Chi0n(mol),
                Descriptors.Chi1n(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.Ipc(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.PEOE_VSA1(mol),
                Descriptors.SMR_VSA1(mol),
                Descriptors.SlogP_VSA1(mol),
                Descriptors.EState_VSA1(mol),
                Descriptors.VSA_EState1(mol),
                Descriptors.MaxAbsEStateIndex(mol),
                Descriptors.MaxEStateIndex(mol),
                Descriptors.MinAbsEStateIndex(mol),
                Descriptors.MinEStateIndex(mol),
                Descriptors.qed(mol),
                # Add more descriptors...
            ]
            
            # Morgan fingerprint bits
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            morgan_bits = [int(x) for x in morgan_fp.ToBitString()]
            
            # Combine descriptors and fingerprint (take first 166 Morgan bits to make 200 total)
            all_features = descriptors + morgan_bits[:166]
            
            # Ensure exactly 200 features
            all_features = all_features[:200]
            while len(all_features) < 200:
                all_features.append(0.0)
            
            return np.array(all_features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error computing molecular descriptors for {smiles}: {e}")
            return np.zeros(200)
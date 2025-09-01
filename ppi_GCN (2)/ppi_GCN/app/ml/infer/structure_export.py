import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.MMCIFIO import MMCIFIO
import json
from typing import Dict, List, Tuple
from pathlib import Path
import tempfile
from loguru import logger
from app.core.config import settings

class StructureExporter:
    """Export protein structures and predictions to various formats"""
    
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
    
    def create_pdb_from_sequence(
        self,
        sequence_a: str,
        sequence_b: str,
        residue_annotations: Dict,
        prediction_id: str
    ) -> str:
        """Create PDB file from sequences with interaction annotations"""
        
        try:
            # Create structure
            structure = Structure.Structure(f"prediction_{prediction_id}")
            model = Model.Model(0)
            structure.add(model)
            
            # Chain A
            chain_a = Chain.Chain('A')
            model.add(chain_a)
            self._add_residues_to_chain(chain_a, sequence_a, 'A', residue_annotations.get('protein_a', {}))
            
            # Chain B
            chain_b = Chain.Chain('B')
            model.add(chain_b)
            self._add_residues_to_chain(chain_b, sequence_b, 'B', residue_annotations.get('protein_b', {}))
            
            # Save PDB
            pdb_path = Path(settings.STORAGE_PATH) / f"prediction_{prediction_id}.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_path))
            
            logger.info(f"PDB saved: {pdb_path}")
            return str(pdb_path)
            
        except Exception as e:
            logger.error(f"PDB creation failed: {e}")
            raise
    
    def _add_residues_to_chain(self, chain: Chain, sequence: str, chain_id: str, annotations: Dict):
        """Add residues to chain with coordinates"""
        
        # Standard amino acid codes
        three_letter = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        residue_scores = annotations.get('residue_scores', [])
        
        for i, aa in enumerate(sequence):
            resname = three_letter.get(aa.upper(), 'UNK')
            residue = Residue.Residue((' ', i + 1, ' '), resname, ' ')
            
            # Generate simple coordinates (extended chain)
            x = i * 3.8  # CA-CA distance
            y = 0.0 if chain_id == 'A' else 10.0  # Separate chains
            z = 0.0
            
            # Add interaction score as B-factor
            b_factor = residue_scores[i] * 100 if i < len(residue_scores) else 0.0
            
            # Add atoms (simplified - just CA)
            ca_atom = Atom.Atom('CA', [x, y, z], b_factor, 1.0, ' ', 'CA', 'C')
            residue.add(ca_atom)
            
            chain.add(residue)
    
    def create_mmcif(self, pdb_path: str, prediction_id: str) -> str:
        """Convert PDB to mmCIF format"""
        try:
            structure = self.pdb_parser.get_structure(f"pred_{prediction_id}", pdb_path)
            
            cif_path = Path(settings.STORAGE_PATH) / f"prediction_{prediction_id}.cif"
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(cif_path))
            
            logger.info(f"mmCIF saved: {cif_path}")
            return str(cif_path)
            
        except Exception as e:
            logger.error(f"mmCIF creation failed: {e}")
            raise
    
    def create_gltf(
        self,
        sequence_a: str,
        sequence_b: str,
        residue_annotations: Dict,
        prediction_id: str
    ) -> str:
        """Create GLTF file for holographic display"""
        
        try:
            # Simplified GLTF creation
            gltf_data = {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0, 1]}],
                "nodes": [
                    {
                        "name": "ProteinA",
                        "mesh": 0,
                        "translation": [0, 0, 0]
                    },
                    {
                        "name": "ProteinB", 
                        "mesh": 1,
                        "translation": [10, 0, 0]
                    }
                ],
                "meshes": [
                    {
                        "name": "ProteinA_Mesh",
                        "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]
                    },
                    {
                        "name": "ProteinB_Mesh", 
                        "primitives": [{"attributes": {"POSITION": 2}, "indices": 3}]
                    }
                ],
                "accessors": [],
                "bufferViews": [],
                "buffers": [],
                "materials": [
                    {
                        "name": "ProteinMaterial",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.8, 0.8, 1.0, 1.0],
                            "metallicFactor": 0.0,
                            "roughnessFactor": 0.9
                        }
                    }
                ]
            }
            
            # Add vertex data (simplified)
            vertices_a = []
            vertices_b = []
            
            for i, aa in enumerate(sequence_a):
                vertices_a.extend([i * 3.8, 0.0, 0.0])  # Simple linear structure
            
            for i, aa in enumerate(sequence_b):
                vertices_b.extend([i * 3.8, 10.0, 0.0])
            
            # Create binary buffer (simplified)
            vertex_data_a = np.array(vertices_a, dtype=np.float32).tobytes()
            vertex_data_b = np.array(vertices_b, dtype=np.float32).tobytes()
            
            gltf_path = Path(settings.STORAGE_PATH) / f"prediction_{prediction_id}.gltf"
            
            with open(gltf_path, 'w') as f:
                json.dump(gltf_data, f, indent=2)
            
            logger.info(f"GLTF saved: {gltf_path}")
            return str(gltf_path)
            
        except Exception as e:
            logger.error(f"GLTF creation failed: {e}")
            raise
    
    def create_surface_mesh(self, pdb_path: str, prediction_id: str) -> str:
        """Create surface mesh from PDB (simplified)"""
        try:
            # This would use MDTraj/ProDy for actual surface generation
            # For now, create a simple mesh representation
            
            mesh_data = {
                "vertices": [],
                "faces": [],
                "normals": [],
                "metadata": {
                    "prediction_id": prediction_id,
                    "surface_type": "molecular_surface"
                }
            }
            
            # Generate simple sphere vertices (placeholder)
            for i in range(100):
                theta = 2 * np.pi * i / 100
                x = 5.0 * np.cos(theta)
                y = 5.0 * np.sin(theta)
                z = 0.0
                mesh_data["vertices"].extend([x, y, z])
                mesh_data["normals"].extend([x/5, y/5, 0])
            
            mesh_path = Path(settings.STORAGE_PATH) / f"surface_{prediction_id}.json"
            with open(mesh_path, 'w') as f:
                json.dump(mesh_data, f)
            
            return str(mesh_path)
            
        except Exception as e:
            logger.error(f"Surface mesh creation failed: {e}")
            raise
# Example:
#   from structures import best_pdb_for_uniprot
#   print(best_pdb_for_uniprot("P00533"))
import requests
from typing import Optional, Tuple, Dict, Any, List

def best_pdb_for_uniprot(accession: str) -> Optional[str]:
    """
    Returns a representative/best PDB ID for a UniProt accession (heuristic: first mapping).
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{accession}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or accession not in data:
        return None
    pdb_map = data[accession].get("PDB", {})
    if not pdb_map:
        return None
    # choose first PDB id deterministically
    pdb_ids = sorted(list(pdb_map.keys()))
    return pdb_ids[0] if pdb_ids else None

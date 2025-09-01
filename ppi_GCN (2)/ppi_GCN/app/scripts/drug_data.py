# Usage (from another script):
#   from drug_data import find_drug_targets
#   targets = find_drug_targets("imatinib")
#   print(targets[:3])
import requests
from typing import Dict, List

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def search_molecule_chembl_ids(drug_name: str, limit: int = 5) -> List[str]:
    url = f"{CHEMBL_BASE}/molecule?molecule_synonyms__icontains={drug_name}&limit={limit}"
    data = _get_json(url)
    return [m["molecule_chembl_id"] for m in data.get("molecules", [])]

def find_drug_targets(drug_name: str, limit: int = 200) -> List[Dict]:
    """
    Returns a list of dicts with keys:
      - molecule_chembl_id
      - target_chembl_id
      - target_pref_name
      - uniprot_accessions (list)
    """
    chembl_ids = search_molecule_chembl_ids(drug_name)
    results: List[Dict] = []
    for cid in chembl_ids:
        # get activities to enumerate target_chembl_id
        acts_url = f"{CHEMBL_BASE}/activity?molecule_chembl_id={cid}&limit={limit}"
        acts = _get_json(acts_url)
        seen_targets = set()
        for act in acts.get("activities", []):
            tid = act.get("target_chembl_id")
            if not tid or tid in seen_targets:
                continue
            seen_targets.add(tid)
            # fetch target to map to UniProt
            tgt_url = f"{CHEMBL_BASE}/target/{tid}"
            tgt = _get_json(tgt_url)
            comps = tgt.get("target_components", [])
            uniprots = []
            for c in comps:
                for xref in c.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        uniprots.append(xref.get("xref_id"))
            results.append({
                "molecule_chembl_id": cid,
                "target_chembl_id": tid,
                "target_pref_name": tgt.get("pref_name"),
                "uniprot_accessions": sorted(list(set([u for u in uniprots if u]))),
            })
    return results

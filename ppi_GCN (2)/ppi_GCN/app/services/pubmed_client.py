import httpx
import asyncio
from typing import List, Dict, Optional
from loguru import logger
import time

class PubChemClient:
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.rate_limit_delay = 0.2
        self.last_request_time = 0
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_compound_by_cid(self, cid: str) -> Optional[Dict]:
        """Get compound information by CID"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/compound/cid/{cid}/property/MolecularWeight,MolecularFormula,CanonicalSMILES,IUPACName/JSON"
                response = await client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    properties = data.get('PropertyTable', {}).get('Properties', [])
                    if properties:
                        return properties[0]
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"PubChem fetch error for CID {cid}: {e}")
            return None
    
    async def search_compounds(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search compounds by name or structure"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/compound/name/{query}/cids/JSON"
                response = await client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get('IdentifierList', {}).get('CID', [])[:max_results]
                    
                    # Fetch details for each CID
                    compounds = []
                    for cid in cids:
                        compound_data = await self.get_compound_by_cid(str(cid))
                        if compound_data:
                            compound_data['cid'] = cid
                            compounds.append(compound_data)
                    
                    return compounds
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"PubChem search error for {query}: {e}")
            return []

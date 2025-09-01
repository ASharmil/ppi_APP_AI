import httpx
import asyncio
from typing import List, Dict, Optional
from loguru import logger
import time

class ChEMBLClient:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.rate_limit_delay = 0.2  # 5 requests per second
        self.last_request_time = 0
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def fetch_molecules(self, offset: int = 0, limit: int = 100) -> List[Dict]:
        """Fetch molecules from ChEMBL"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/molecule"
                params = {
                    "format": "json",
                    "offset": offset,
                    "limit": limit,
                    "molecule_type": "Small molecule"
                }
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('molecules', [])
                else:
                    logger.error(f"ChEMBL API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"ChEMBL fetch error: {e}")
            return []
    
    async def get_drug_targets(self, chembl_id: str) -> List[Dict]:
        """Get targets for a specific drug"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/drug_mechanism"
                params = {
                    "format": "json",
                    "molecule_chembl_id": chembl_id
                }
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('drug_mechanisms', [])
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"ChEMBL targets fetch error for {chembl_id}: {e}")
            return []

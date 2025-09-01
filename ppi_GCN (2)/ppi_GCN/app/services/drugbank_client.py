import httpx
import asyncio
from typing import List, Dict, Optional
from loguru import logger
import time
from app.core.config import settings

class DrugBankClient:
    def __init__(self):
        self.base_url = "https://go.drugbank.com/api/v1"
        self.api_key = settings.DRUGBANK_API_KEY
        self.rate_limit_delay = 1.0  # 1 request per second
        self.last_request_time = 0
    
    async def _rate_limit(self):
        await asyncio.sleep(self.rate_limit_delay)
    
    async def fetch_drugs(self, offset: int = 0, limit: int = 100) -> List[Dict]:
        """Fetch drugs from DrugBank API"""
        if not self.api_key:
            logger.warning("DrugBank API key not configured")
            return []
        
        await self._rate_limit()
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/drugs"
                params = {"offset": offset, "limit": limit}
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('drugs', [])
                else:
                    logger.error(f"DrugBank API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"DrugBank fetch error: {e}")
            return []
    
    async def get_drug_details(self, drugbank_id: str) -> Optional[Dict]:
        """Get detailed information for a specific drug"""
        if not self.api_key:
            return None
        
        await self._rate_limit()
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/drugs/{drugbank_id}"
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"DrugBank detail fetch error for {drugbank_id}: {e}")
            return None

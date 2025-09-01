import httpx
import asyncio
from typing import Optional, Dict, List
from loguru import logger
import time

class UniProtClient:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
        self.rate_limit_delay = 0.1  # 10 requests per second
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Ensure rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_sequence(self, uniprot_id: str) -> Optional[str]:
        """Fetch amino acid sequence from UniProt"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/uniprotkb/{uniprot_id}.fasta"
                response = await client.get(url)
                
                if response.status_code == 200:
                    fasta_content = response.text
                    # Parse FASTA
                    lines = fasta_content.strip().split('\n')
                    if len(lines) > 1:
                        sequence = ''.join(lines[1:])
                        logger.info(f"Retrieved sequence for {uniprot_id}: {len(sequence)} residues")
                        return sequence
                else:
                    logger.warning(f"UniProt request failed for {uniprot_id}: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"UniProt client error for {uniprot_id}: {e}")
            return None
    
    async def get_protein_info(self, uniprot_id: str) -> Optional[Dict]:
        """Fetch detailed protein information"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/uniprotkb/{uniprot_id}.json"
                response = await client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                        'organism': data.get('organism', {}).get('scientificName', ''),
                        'function': data.get('comments', [{}])[0].get('texts', [{}])[0].get('value', '') if data.get('comments') else '',
                        'gene_name': data.get('genes', [{}])[0].get('geneName', {}).get('value', '') if data.get('genes') else ''
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"UniProt info fetch error for {uniprot_id}: {e}")
            return None

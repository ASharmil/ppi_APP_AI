import httpx
import asyncio
from typing import Optional, Dict
from loguru import logger
import time

class PDBClient:
    def __init__(self):
        self.base_url = "https://data.rcsb.org/rest/v1"
        self.rate_limit_delay = 0.1
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Ensure rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_sequence(self, pdb_id: str, chain: str = 'A') -> Optional[str]:
        """Fetch sequence from PDB"""
        await self._rate_limit()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/chain/{chain}"
                response = await client.get(url)
                
                if response.status_code == 200:
                    fasta_content = response.text
                    lines = fasta_content.strip().split('\n')
                    if len(lines) > 1:
                        sequence = ''.join(lines[1:])
                        logger.info(f"Retrieved PDB sequence for {pdb_id}:{chain}: {len(sequence)} residues")
                        return sequence
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"PDB client error for {pdb_id}: {e}")
            return None
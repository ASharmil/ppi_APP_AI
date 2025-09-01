import redis
import json
import pickle
from typing import Any, Optional
from app.core.config import settings
from loguru import logger

class CacheService:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration"""
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, expire, serialized)
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
    
    def generate_protein_key(self, protein_id: str) -> str:
        return f"protein:sequence:{protein_id}"
    
    def generate_drug_key(self, drug_id: str, source: str) -> str:
        return f"drug:{source}:{drug_id}"

cache_service = CacheService()
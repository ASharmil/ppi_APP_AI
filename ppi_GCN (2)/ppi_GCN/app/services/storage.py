from pathlib import Path
import aiofiles
import hashlib
from typing import BinaryIO, Optional
from app.core.config import settings
from loguru import logger
import urllib.parse

class StorageService:
    def __init__(self):
        self.storage_path = Path(settings.STORAGE_PATH)
        self.storage_path.mkdir(exist_ok=True, parents=True)
    
    async def save_file(self, content: bytes, filename: str, subfolder: str = "") -> str:
        """Save file to storage"""
        try:
            if subfolder:
                folder_path = self.storage_path / subfolder
                folder_path.mkdir(exist_ok=True, parents=True)
                file_path = folder_path / filename
            else:
                file_path = self.storage_path / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"File save error: {e}")
            raise
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """Get file content"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"File read error: {e}")
            return None
    
    def generate_signed_url(self, file_path: str, expiry_hours: int = 24) -> str:
        """Generate signed URL for file access (simplified)"""
        # In production, use proper signed URLs with cloud storage
        relative_path = Path(file_path).relative_to(self.storage_path)
        return f"/api/v1/files/{urllib.parse.quote(str(relative_path))}"
    
    def get_file_hash(self, content: bytes) -> str:
        """Generate file hash"""
        return hashlib.sha256(content).hexdigest()

storage_service = StorageService()
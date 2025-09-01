import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

class HologramStreamer:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.is_streaming = False
    
    async def connect(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Hologram client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        self.active_connections.discard(websocket)
        logger.info(f"Hologram client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_prediction(self, prediction_data: Dict):
        """Broadcast prediction to all connected clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": "prediction_update",
            "data": {
                "interaction_score": prediction_data.get("interaction_score"),
                "binding_affinity": prediction_data.get("binding_affinity"),
                "confidence": prediction_data.get("confidence"),
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        
        await self._broadcast(message)
    
    async def broadcast_structure(self, structure_data: Dict):
        """Broadcast 3D structure data"""
        message = {
            "type": "structure_update",
            "data": {
                "pdb_url": structure_data.get("pdb_url"),
                "gltf_url": structure_data.get("gltf_url"),
                "residue_annotations": structure_data.get("residue_annotations"),
                "surface_mesh": structure_data.get("surface_mesh")
            }
        }
        
        await self._broadcast(message)
    
    async def broadcast_progress(self, job_id: str, progress: int, status: str):
        """Broadcast training/prediction progress"""
        message = {
            "type": "progress_update",
            "data": {
                "job_id": job_id,
                "progress": progress,
                "status": status
            }
        }
        
        await self._broadcast(message)
    
    async def _broadcast(self, message: Dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = set()
        json_message = json.dumps(message)
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json_message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

hologram_streamer = HologramStreamer()
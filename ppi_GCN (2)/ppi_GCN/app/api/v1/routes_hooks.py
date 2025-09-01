"""
Arduino and Hologram device hooks
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from app.core.security import get_current_user
from app.db.session import get_db
from app.db.models.user import User
from app.services.hooks.arduino import ArduinoPublisher
from app.services.hooks.hologram_ws import HologramStreamer
from app.core.blockchain import log_blockchain_event

router = APIRouter()

class ArduinoMessage(BaseModel):
    port: str
    message: dict
    dry_run: bool = True

class ArduinoResponse(BaseModel):
    status: str
    port: str
    message_sent: bool
    error: Optional[str] = None

@router.post("/arduino/publish", response_model=ArduinoResponse)
async def publish_to_arduino(
    message_data: ArduinoMessage,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Publish structured data to Arduino device"""
    
    try:
        publisher = ArduinoPublisher()
        
        result = await publisher.publish(
            port=message_data.port,
            message=message_data.message,
            dry_run=message_data.dry_run
        )
        
        # Log to blockchain
        await log_blockchain_event(
            user_address=str(current_user.id),
            action="hooks.arduino",
            ref_id=message_data.port
        )
        
        return ArduinoResponse(
            status="success" if result["sent"] else "failed",
            port=message_data.port,
            message_sent=result["sent"],
            error=result.get("error")
        )
        
    except Exception as e:
        return ArduinoResponse(
            status="error",
            port=message_data.port,
            message_sent=False,
            error=str(e)
        )

@router.websocket("/hologram/stream")
async def hologram_websocket(
    websocket: WebSocket,
    prediction_id: Optional[int] = None
):
    """WebSocket stream for hologram device"""
    
    await websocket.accept()
    
    try:
        streamer = HologramStreamer()
        
        if prediction_id:
            # Stream specific prediction results
            async for data in streamer.stream_prediction_results(prediction_id):
                await websocket.send_text(json.dumps(data))
                await asyncio.sleep(0.1)  # Rate limiting
        else:
            # Stream live data
            async for data in streamer.stream_live_data():
                await websocket.send_text(json.dumps(data))
                await asyncio.sleep(0.5)
                
    except WebSocketDisconnect:
        print("Hologram WebSocket disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()

@router.get("/arduino/simulate")
async def simulate_arduino_data(
    current_user: User = Depends(get_current_user)
):
    """Simulate Arduino data for testing"""
    
    simulation_data = {
        "residues": [
            {"position": i, "amino_acid": "ALA", "interaction_prob": 0.1 + (i % 10) * 0.1}
            for i in range(1, 21)
        ],
        "binding_affinity": 0.85,
        "docking_score": -12.5,
        "timestamp": "2025-08-31T12:00:00Z"
    }
    
    return {"status": "simulation", "data": simulation_data}

@router.get("/hologram/simulate")
async def simulate_hologram_data(
    current_user: User = Depends(get_current_user)
):
    """Simulate hologram data for testing"""
    
    simulation_data = {
        "structure_metadata": {
            "atoms": 1248,
            "residues": 156,
            "chains": 2,
            "resolution": "2.1A"
        },
        "interaction_sites": [
            {"chain": "A", "residue": 42, "type": "binding"},
            {"chain": "B", "residue": 78, "type": "catalytic"}
        ],
        "surface_mesh": {
            "vertices": 5000,
            "faces": 10000,
            "format": "gltf"
        }
    }
    
    return {"status": "simulation", "data": simulation_data}

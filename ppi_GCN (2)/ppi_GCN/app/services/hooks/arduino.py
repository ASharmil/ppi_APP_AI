import serial
import json
import asyncio
from typing import Dict, Optional
from loguru import logger
from app.core.config import settings

class ArduinoPublisher:
    def __init__(self):
        self.port = settings.ARDUINO_PORT
        self.baud_rate = settings.ARDUINO_BAUD_RATE
        self.connection = None
        self.is_simulation = True  # Default to simulation mode
    
    async def connect(self, simulation: bool = False):
        """Connect to Arduino or start simulation"""
        self.is_simulation = simulation
        
        if not simulation:
            try:
                self.connection = serial.Serial(self.port, self.baud_rate, timeout=1)
                await asyncio.sleep(2)  # Wait for Arduino to reset
                logger.info(f"Arduino connected on {self.port}")
                return True
            except Exception as e:
                logger.warning(f"Arduino connection failed: {e}, falling back to simulation")
                self.is_simulation = True
        
        logger.info("Arduino simulation mode enabled")
        return True
    
    async def publish_prediction(self, prediction_data: Dict):
        """Publish prediction results to Arduino"""
        try:
            message = {
                "type": "prediction",
                "data": {
                    "interaction_score": prediction_data.get("interaction_score", 0),
                    "binding_affinity": prediction_data.get("binding_affinity", 0),
                    "confidence": prediction_data.get("confidence", 0),
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
            
            if self.is_simulation:
                logger.info(f"Arduino simulation - would send: {json.dumps(message, indent=2)}")
                return True
            
            if self.connection and self.connection.is_open:
                json_message = json.dumps(message)
                self.connection.write(json_message.encode() + b'\n')
                self.connection.flush()
                logger.info(f"Sent to Arduino: {json_message}")
                return True
            else:
                logger.error("Arduino not connected")
                return False
                
        except Exception as e:
            logger.error(f"Arduino publish error: {e}")
            return False
    
    async def publish_residue_data(self, residue_annotations: Dict):
        """Publish residue-level interaction data"""
        try:
            message = {
                "type": "residues",
                "data": {
                    "protein_a_scores": residue_annotations.get("protein_a", {}).get("residue_scores", []),
                    "protein_b_scores": residue_annotations.get("protein_b", {}).get("residue_scores", []),
                    "high_confidence_a": residue_annotations.get("protein_a", {}).get("high_confidence_residues", []),
                    "high_confidence_b": residue_annotations.get("protein_b", {}).get("high_confidence_residues", [])
                }
            }
            
            if self.is_simulation:
                logger.info(f"Arduino simulation - residue data: {len(message['data']['protein_a_scores'])} + {len(message['data']['protein_b_scores'])} residues")
                return True
            
            if self.connection and self.connection.is_open:
                json_message = json.dumps(message)
                self.connection.write(json_message.encode() + b'\n')
                self.connection.flush()
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Arduino residue publish error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Arduino"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Arduino disconnected")

arduino_publisher = ArduinoPublisher()

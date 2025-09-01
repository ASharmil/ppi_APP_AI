from web3 import Web3
from solcx import compile_source, install_solc
from app.core.config import settings
from loguru import logger
import json
import os

class BlockchainLogger:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(settings.ETHEREUM_RPC_URL))
        self.contract = None
        self.account = None
        
        if settings.PRIVATE_KEY:
            self.account = self.w3.eth.account.from_key(settings.PRIVATE_KEY)
        
        if settings.CONTRACT_ADDRESS:
            self._load_contract()
    
    def _load_contract(self):
        try:
            with open("contracts/PPILedger.json", "r") as f:
                contract_data = json.load(f)
                self.contract = self.w3.eth.contract(
                    address=settings.CONTRACT_ADDRESS,
                    abi=contract_data["abi"]
                )
        except Exception as e:
            logger.warning(f"Could not load contract: {e}")
    
    async def log_event(self, user_address: str, action: str, ref_id: str):
        if not self.contract or not self.account:
            logger.warning("Blockchain not configured, skipping log")
            return None
        
        try:
            tx = self.contract.functions.logEvent(
                user_address, action, ref_id
            ).build_transaction({
                'from': self.account.address,
                'gas': 100000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Blockchain event logged: {action} - {tx_hash.hex()}")
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Blockchain logging failed: {e}")
            return None

blockchain_logger = BlockchainLogger()
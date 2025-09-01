"""
Blockchain contract deployment and management utilities.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from web3 import Web3
from web3.contract import Contract
from solcx import compile_source, install_solc, set_solc_version
import logging

logger = logging.getLogger(__name__)


class ContractDeployer:
    """Handles deployment and interaction with PPILedger smart contract."""
    
    def __init__(self, rpc_url: str = "http://localhost:8545", private_key: Optional[str] = None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key
        self.account = None
        
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
        else:
            # Use first account from local node (development)
            if self.w3.eth.accounts:
                self.account = self.w3.eth.accounts[0]
        
        self.contracts_dir = Path(__file__).parent
        self.abi_file = self.contracts_dir / "abi.json"
        self.address_file = self.contracts_dir / "address.txt"
        
        # Install Solidity compiler if needed
        try:
            install_solc("0.8.19")
            set_solc_version("0.8.19")
        except Exception as e:
            logger.warning(f"Could not install solc: {e}")
    
    def compile_contract(self) -> Dict[str, Any]:
        """Compile the PPILedger contract."""
        contract_path = self.contracts_dir / "PPILedger.sol"
        
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")
        
        with open(contract_path, 'r') as f:
            contract_source = f.read()
        
        try:
            compiled_sol = compile_source(contract_source)
            contract_interface = compiled_sol['<stdin>:PPILedger']
            
            return {
                'abi': contract_interface['abi'],
                'bytecode': contract_interface['bin']
            }
        except Exception as e:
            logger.error(f"Contract compilation failed: {e}")
            raise
    
    def deploy_contract(self) -> tuple[str, Dict[str, Any]]:
        """Deploy the PPILedger contract."""
        if not self.w3.is_connected():
            raise ConnectionError("Not connected to blockchain network")
        
        if not self.account:
            raise ValueError("No account available for deployment")
        
        logger.info("Compiling contract...")
        contract_data = self.compile_contract()
        
        # Create contract instance
        contract = self.w3.eth.contract(
            abi=contract_data['abi'],
            bytecode=contract_data['bytecode']
        )
        
        # Get account address
        if isinstance(self.account, str):
            account_address = self.account
        else:
            account_address = self.account.address
        
        # Build transaction
        logger.info("Deploying contract...")
        constructor_tx = contract.constructor().build_transaction({
            'from': account_address,
            'gas': 3000000,
            'gasPrice': self.w3.to_wei('20', 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(account_address)
        })
        
        # Sign and send transaction
        if self.private_key:
            signed_tx = self.w3.eth.account.sign_transaction(constructor_tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            # For local development with unlocked accounts
            tx_hash = self.w3.eth.send_transaction(constructor_tx)
        
        # Wait for deployment
        logger.info(f"Waiting for deployment transaction: {tx_hash.hex()}")
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise RuntimeError("Contract deployment failed")
        
        contract_address = tx_receipt.contractAddress
        logger.info(f"Contract deployed at: {contract_address}")
        
        # Save ABI and address
        self.save_contract_data(contract_address, contract_data['abi'])
        
        return contract_address, contract_data['abi']
    
    def save_contract_data(self, address: str, abi: Dict[str, Any]):
        """Save contract ABI and address to files."""
        # Save ABI
        with open(self.abi_file, 'w') as f:
            json.dump(abi, f, indent=2)
        
        # Save address
        with open(self.address_file, 'w') as f:
            f.write(address)
        
        logger.info(f"Contract data saved: ABI -> {self.abi_file}, Address -> {self.address_file}")
    
    def load_contract_data(self) -> tuple[str, Dict[str, Any]]:
        """Load contract ABI and address from files."""
        if not self.abi_file.exists() or not self.address_file.exists():
            raise FileNotFoundError("Contract data files not found. Deploy contract first.")
        
        with open(self.abi_file, 'r') as f:
            abi = json.load(f)
        
        with open(self.address_file, 'r') as f:
            address = f.read().strip()
        
        return address, abi
    
    def get_contract(self) -> Contract:
        """Get contract instance for interaction."""
        address, abi = self.load_contract_data()
        return self.w3.eth.contract(address=address, abi=abi)
    
    def authorize_user(self, user_address: str) -> str:
        """Authorize a user to log events."""
        contract = self.get_contract()
        
        if isinstance(self.account, str):
            account_address = self.account
        else:
            account_address = self.account.address
        
        tx = contract.functions.authorizeUser(user_address).build_transaction({
            'from': account_address,
            'gas': 100000,
            'gasPrice': self.w3.to_wei('20', 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(account_address)
        })
        
        if self.private_key:
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.w3.eth.send_transaction(tx)
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex()
    
    def get_logs(self, user_address: Optional[str] = None, action: Optional[str] = None, 
                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs from the contract."""
        contract = self.get_contract()
        
        try:
            if user_address:
                logs = contract.functions.getLogsByUser(
                    Web3.to_checksum_address(user_address), 0, limit
                ).call()
            elif action:
                logs = contract.functions.getLogsByAction(action, 0, limit).call()
            else:
                logs = contract.functions.getRecentLogs(limit).call()
            
            # Convert to dict format
            formatted_logs = []
            for log in logs:
                formatted_logs.append({
                    'id': log[0],
                    'user': log[1],
                    'action': log[2],
                    'refId': log[3],
                    'metadata': log[4],
                    'timestamp': log[5],
                    'txHash': log[6].hex() if isinstance(log[6], bytes) else log[6]
                })
            
            return formatted_logs
        
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get contract statistics."""
        contract = self.get_contract()
        
        try:
            stats = contract.functions.getStats().call()
            return {
                'total_events': stats[0],
                'total_users': stats[1],
                'timestamp': stats[2]
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


def deploy_ppi_ledger(rpc_url: str = "http://localhost:8545", private_key: Optional[str] = None) -> str:
    """Deploy PPILedger contract and return address."""
    deployer = ContractDeployer(rpc_url, private_key)
    address, abi = deployer.deploy_contract()
    return address


def get_contract_instance(rpc_url: str = "http://localhost:8545") -> Contract:
    """Get deployed contract instance."""
    deployer = ContractDeployer(rpc_url)
    return deployer.get_contract()


if __name__ == "__main__":
    import argparse
    from app.core.config import settings
    
    parser = argparse.ArgumentParser(description="Deploy or interact with PPILedger contract")
    parser.add_argument("--deploy", action="store_true", help="Deploy the contract")
    parser.add_argument("--authorize", type=str, help="Authorize user address")
    parser.add_argument("--logs", action="store_true", help="Get recent logs")
    parser.add_argument("--stats", action="store_true", help="Get contract stats")
    parser.add_argument("--rpc-url", default="http://localhost:8545", help="RPC URL")
    parser.add_argument("--private-key", help="Private key for transactions")
    
    args = parser.parse_args()
    
    try:
        if args.deploy:
            print("Deploying PPILedger contract...")
            address = deploy_ppi_ledger(args.rpc_url, args.private_key)
            print(f"Contract deployed at: {address}")
        
        elif args.authorize:
            deployer = ContractDeployer(args.rpc_url, args.private_key)
            tx_hash = deployer.authorize_user(args.authorize)
            print(f"User authorized. Transaction: {tx_hash}")
        
        elif args.logs:
            deployer = ContractDeployer(args.rpc_url)
            logs = deployer.get_logs(limit=20)
            print(f"Recent logs ({len(logs)} entries):")
            for log in logs:
                print(f"  {log['timestamp']}: {log['action']} by {log['user']} (ref: {log['refId']})")
        
        elif args.stats:
            deployer = ContractDeployer(args.rpc_url)
            stats = deployer.get_stats()
            print(f"Contract stats: {stats}")
        
        else:
            print("Use --deploy, --authorize <address>, --logs, or --stats")
    
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
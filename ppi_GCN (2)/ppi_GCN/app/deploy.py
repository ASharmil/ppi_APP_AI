import json
from pathlib import Path

from solcx import compile_source, install_solc
from web3 import Web3

from app.core.config import settings
from app.core.logging_conf import logger, setup_logging


def deploy_contract():
    """
    Compiles and deploys the PPILedger.sol contract.
    Saves the contract address and ABI to a build file.
    """
    setup_logging()
    logger.info("Starting contract deployment...")

    # --- 1. Compile Solidity Contract ---
    contract_path = Path(__file__).parent / "PPILedger.sol"
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)

    logger.info(f"Compiling contract: {contract_path}")
    try:
        install_solc(version="0.8.0")
    except Exception as e:
        logger.warning(f"solc 0.8.0 might already be installed: {e}")

    with open(contract_path, "r") as f:
        source = f.read()

    compiled_sol = compile_source(source, output_values=["abi", "bin"])
    contract_id, contract_interface = compiled_sol.popitem()
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]

    # --- 2. Connect to Blockchain and Deploy ---
    w3 = Web3(Web3.HTTPProvider(settings.BLOCKCHAIN_RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Could not connect to blockchain at {settings.BLOCKCHAIN_RPC_URL}")

    w3.eth.default_account = w3.eth.accounts[0]
    logger.info(f"Using account: {w3.eth.default_account}")

    PPILedger = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = PPILedger.constructor().transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress

    logger.info(f"Contract deployed successfully! Address: {contract_address}")

    # --- 3. Save ABI and Address ---
    with open(build_dir / "PPILedger.json", "w") as f:
        json.dump({"address": contract_address, "abi": abi}, f, indent=4)
    logger.info(f"Contract info saved to {build_dir / 'PPILedger.json'}")


if __name__ == "__main__":
    deploy_contract()


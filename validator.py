import argparse 
import logging

from automl.validators import ValidatorFactory
from neurons.btt_connector import BittensorNetwork
from neurons.chain_manager import ChainMultiAddressStore
from neurons.hf_manager import HFManager
from config import config 


def setup_logging(log_file='validator.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main(config):
    # Initialize Bittensor Network
    setup_logging()
    logging.info(f"Starting miner of type: {config.Validator.validator_type}")
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)

    config.bittensor_network = BittensorNetwork

    # Initialize Chain Manager and HF Manager
    config.chain_manager =  ChainMultiAddressStore(BittensorNetwork.subtensor, bt_config.netuid, BittensorNetwork.wallet)
    
    config.hf_manager = HFManager(
        local_dir=".",
        hf_token=config.hf_token,
        gene_repo_id=config.gene_repo,
        device=config.device
    )
    
    # Create and start validator
    validator = ValidatorFactory.get_validator(config)
    validator.measure_baseline()
    logging.info("Starting periodic validation")
    validator.start_periodic_validation()

if __name__ == "__main__":
    # config = {
    #     'bittensor_config': config.bittensor_config,
    #     'netuid': config.netuid,
    #     'hf_token': config.hf_token,
    #     'my_repo_id': config.my_repo_id,
    #     'averaged_model_repo_id': config.averaged_model_repo_id,
    #     'device': config.device,
    #     'validation_interval': config.validation_interval,
    #     'hf_repo': config.hf_repo
    # }
    
    validator_type = "activation"  # Change this to "loss" as needed
    main(config)
from automl.validators import ValidatorFactory
from neurons.btt_connector import BittensorNetwork
from neurons.chain_manager import ChainMultiAddressStore
from neurons.hf_manager import HFManager
from neurons.config import Config

def main(validator_type, config):
    # Initialize Bittensor Network
    bt_config = config['bittensor_config']
    BittensorNetwork.initialize(bt_config)
    
    # Initialize Chain Manager and HF Manager
    chain_manager = ChainMultiAddressStore(BittensorNetwork.subtensor, config['netuid'], BittensorNetwork.wallet)
    hf_manager = HFManager(
        local_dir=".",
        hf_token=config['hf_token'],
        my_repo_id=config['my_repo_id'],
        averaged_model_repo_id=config['averaged_model_repo_id'],
        device=config['device']
    )
    
    # Update config with necessary components
    config.update({
        'chain_manager': chain_manager,
        'hf_manager': hf_manager,
        'bittensor_network': BittensorNetwork
    })
    
    # Create and start validator
    validator = ValidatorFactory.get_validator(validator_type, config)
    validator.measure_baseline()
    validator.start_periodic_validation()

if __name__ == "__main__":
    config = {
        'bittensor_config': Config.bittensor_config,
        'netuid': Config.netuid,
        'hf_token': Config.hf_token,
        'my_repo_id': Config.my_repo_id,
        'averaged_model_repo_id': Config.averaged_model_repo_id,
        'device': Config.device,
        'validation_interval': Config.validation_interval,
        'hf_repo': Config.hf_repo
    }
    
    validator_type = "activation"  # Change this to "loss" as needed
    main(validator_type, config)
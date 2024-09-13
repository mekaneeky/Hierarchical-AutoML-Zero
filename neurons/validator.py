import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from huggingface_hub import HfApi
import json
import logging
from typing import List, Dict, Tuple

from automl.models import EvolvableNN
from automl.function_decoder import FunctionDecoder
from automl.gene_io import import_gene_from_json
from validator_class import AutoMLValidator

from btt_connector import BittensorNetwork
from chain_manager import ChainMultiAddressStore
from hf_manager import HFManager
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data() -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    return train_loader, val_loader

def main():
    # Initialize Bittensor Network
    bt_config = bt.config(Config.wallet_name)
    bt_config.netuid = Config.netuid
    bt_config.wallet.hotkey = Config.wallet_hotkey
    bt_config.subtensor.chain_endpoint = Config.chain_endpoint
    BittensorNetwork.initialize(bt_config)
    
    # Initialize Chain Manager and HF Manager
    chain_manager = ChainMultiAddressStore(BittensorNetwork.subtensor, Config.netuid, BittensorNetwork.wallet)
    hf_manager = HFManager(
        local_dir=".",
        hf_token=Config.hf_token,
        my_repo_id=Config.my_repo_id,
        averaged_model_repo_id=Config.averaged_model_repo_id,
        device=Config.device
    )
    
    # Load MNIST data
    val_loader = load_mnist_data()
    
    # Initialize AutoMLValidator
    validator = AutoMLValidator(
        device=Config.device,
        model=None,  # We don't need a predefined model for AutoML
        optimizer=None,  # We don't need an optimizer for validation
        data_loader=val_loader,
        bittensor_network=BittensorNetwork,
        chain_manager=chain_manager,
        hf_manager=hf_manager,
        interval=Config.validation_interval,
        local_gradient_dir=Config.local_gradient_dir
    )
    
    # Start periodic validation
    validator.start_periodic_validation()

if __name__ == "__main__":
    main()
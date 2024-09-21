# Miner and Validator Tutorial

This tutorial will guide you through the process of setting up and running miners and validators for the AutoML network.

## Miner Setup

### Prerequisite

If you don't have a key on the bittensor network refer to [this](https://docs.bittensor.com/getting-started/wallets). You can refer to [taostats.io](https://taostats.io) to find out where to buy TAO to cover your registration cost.

### 1. Register Your Miner

First, you need to register your miner on the network:

```
btcli s register
```

### 2. Configure Your Miner

Edit the `config.py` file to set the following configurations:

- `config.Bittensor.netuid`: Set the network UID (100 for testnet, 38 for mainnet)
- `config.Bittensor.wallet_name`: Set your wallet name
- `config.Bittensor.wallet_hotkey`: Set your wallet hotkey
- `config.Bittensor.network`: Set to "test" for testnet or "finney" for mainnet
- `config.Bittensor.chain_endpoint`: Edit if using your own subtensor node
- `config.hf_token`: Set your Hugging Face token
- `config.gene_repo`: Set your Hugging Face repository name for storing genes

Example:

```python
class config:
    hf_token = "your_huggingface_token"
    gene_repo = "your_username/your_repo_name"

    class Bittensor:
        netuid = 100
        wallet_name = "your_wallet_name"
        wallet_hotkey = "your_wallet_hotkey"
        network = "test"  # or "finney" for mainnet
```

### 3. Register Metadata

Run the following script to register your Hugging Face repository to the chain:

```
python register_miner.py
```

### 4. Configure Miner Settings

In `config.py`, adjust the `Miner` class settings as needed:

```python
class Miner:
    population_size = 100
    num_meta_levels = 1
    genome_length = 5
    tournament_size = 7
    generations = 1000
    generation_iters = 100
    miner_type = "loss"  # or "activation" or "simple"
```

### 5. Run the Miner

Execute the miner script:

```
python miner.py
```

## Validator Setup

### 1. Register Your Validator

Register your validator on the network:

```
btcli s register
```

### 2. Configure Your Validator

Edit the `config.py` file to set the following configurations:

- `config.Bittensor.netuid`: Set the network UID
- `config.Bittensor.wallet_name`: Set your wallet name
- `config.Bittensor.wallet_hotkey`: Set your wallet hotkey
- `config.Bittensor.network`: Set to "test" for testnet or "finney" for mainnet
- `config.hf_token`: Set your Hugging Face token

Example:

```python
class config:
    hf_token = "your_huggingface_token"

    class Bittensor:
        netuid = 100
        wallet_name = "your_wallet_name"
        wallet_hotkey = "your_wallet_hotkey"
        network = "test"  # or "finney" for mainnet
```

### 3. Configure Validator Settings

In `config.py`, adjust the `Validator` class settings:

```python
class Validator:
    validation_interval = 300  # Interval between validations in seconds
    validator_type = "loss"  # or "activation"
    top_k = 10  # Number of top miners to distribute scores to
    min_score = 0.0  # Minimum score for miners not in the top-k
```

### 4. Run the Validator

Execute the validator script:

```
python validator.py
```

## Additional Notes

- Ensure you have the required dependencies installed. You may need to run `pip install -r requirements.txt` (if a requirements file is provided).
- The `metrics_file` in `config.py` specifies where performance metrics will be saved.
- For both miners and validators, make sure you have sufficient balance in your wallet to pay for transaction fees.
- Monitor the console output and log files (`miner.log` for miners, `validator.log` for validators) for any errors or important information.
- The `device` setting in `config.py` determines whether to use CPU or GPU. Set it to "cuda" if you want to use a GPU.

Remember to keep your wallet information and Hugging Face token secure and never share them publicly.
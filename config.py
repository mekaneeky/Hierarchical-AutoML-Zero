import bittensor as bt

class config:
    # Global Configuration
    device = "cpu"  # Use "cuda" for GPU, "cpu" for CPU
    hf_token = "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"
    gene_repo = "mekaneeky/testing-repo-4"
    metrics_file = "metrics.csv"
    # Hugging Face Configuration

    class Bittensor:
        netuid = 100
        wallet_name = "test_wallet_1"
        wallet_hotkey = "test_hot_1"
        path = "~/.bittensor/wallets/"
        network = "test"  # or "finney" for mainnet
        epoch_length = 100
        #subtensor_chain_endpoint = bt.__finney_entrypoint__ #"ws://127.0.0.1:9944"  # local subtensor

    @classmethod
    def get_bittensor_config(cls):
        bt_config = bt.config()
        bt_config.wallet = bt.config()
        bt_config.subtensor = bt.config()
        bt_config.netuid = cls.Bittensor.netuid
        bt_config.wallet.name = cls.Bittensor.wallet_name
        bt_config.wallet.hotkey = cls.Bittensor.wallet_hotkey
        bt_config.subtensor.network = cls.Bittensor.network
        bt_config.epoch_length = cls.Bittensor.epoch_length
        #bt_config.subtensor.chain_endpoint = cls.Bittensor.subtensor_chain_endpoint
        return bt_config

    # Miner Configuration
    class Miner:
        #TODO limit validator memory allowed to prevent DOS attacks 
        population_size = 100
        num_meta_levels = 1
        genome_length = 5
        tournament_size = 7
        generations = 1000
        generation_iters = 100
        num_scalars = 5
        num_vectors = 5
        num_tensors = 5
        scalar_size = 1
        vector_size = (128,)
        tensor_size = (128, 128)
        input_addresses = [5, 6]
        output_addresses = [7]
        miner_type = "loss"
        migration_server_url = None
        migration_interval = 0
        pool_url = "http://127.0.0.1:5000"


    # Validator Configuration
    class Validator:
        validation_interval = 300  # Interval between validations in seconds
        validator_type = "loss"
        top_k = 10  # Number of top miners to distribute scores to
        min_score = 0.0  # Minimum score for miners not in the top-k


    
    # MNIST Configuration (if needed)
    BATCH_SIZE = 1
    INPUT_SIZE = 28 * 28
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    TRAIN_BATCHES = 10

    NUM_GENERATIONS = 1
    GENERATION_ITERS = 2

    MIN_FITNESS = -9999

    # Paths
    data_dir = "./data"
    local_gradient_dir = "./local_gradients"

    # Logging
    log_level = "INFO"

    # Chain Configuration
    chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"


# config.py

class Config:
    # Bittensor Network Configuration
    netuid = 1  # Replace with your subnet's netuid
    wallet_name = "default"  # Replace with your wallet name
    wallet_hotkey = "default"  # Replace with your wallet's hotkey

    # Device Configuration
    device = "cuda"  # Use "cuda" for GPU, "cpu" for CPU

    # Hugging Face Configuration
    hf_token = "your_huggingface_token_here"
    my_repo_id = "your_username/your_repo_name"
    averaged_model_repo_id = "subnet_name/averaged_model_repo"

    # Validator Configuration
    validation_interval = 300  # Interval between validations in seconds

    # AutoML Configuration
    population_size = 50
    num_meta_levels = 1
    genome_length = 20
    tournament_size = 20

    # MNIST Configuration
    
    NUM_SCALARS = 5
    NUM_VECTORS = 5
    NUM_TENSORS = 5
    SCALAR_SIZE = 1
    VECTOR_SIZE = (128,)
    TENSOR_SIZE = (128,128)
    
    POPULATION_SIZE = 2
    NUM_META_LEVELS = 1
    GENOME_LENGTH = 20
    TOURNAMENT_SIZE = 1
    
    BATCH_SIZE = 1
    INPUT_SIZE = 28*28
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

    # Additional configurations can be added here as needed
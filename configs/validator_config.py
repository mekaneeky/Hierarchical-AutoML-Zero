class ValidatorConfig:
    validation_interval = 300  # Interval between validations in seconds
    validator_type = "loss"
    top_k = 10  # Number of top miners to distribute scores to
    min_score = 0.0  # Minimum score for miners not in the top-k
    seed = 42
    
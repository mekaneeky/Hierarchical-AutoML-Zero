from automl.miners import MinerFactory
from automl.unpacker import HierarchicalGenomeUnpacker

def main(miner_type, config):
    miner = MinerFactory.get_miner(miner_type, config)
    best_genome = miner.mine()

    unpacker = HierarchicalGenomeUnpacker()
    print(unpacker.unpack_function_genome(best_genome))
    print(best_genome.gene)
    print(f"Best genome fitness: {best_genome.fitness:.4f}")
    print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
    print(f"Improvement over baseline: {best_genome.fitness - miner.baseline_accuracy:.4f}")
    return best_genome

if __name__ == "__main__":
    config = {
        'population_size': 100,
        'num_meta_levels': 1,
        'genome_length': 5,
        'tournament_size': 70,
        'generations': 1000,
        'generation_iters': 1000,
        'num_scalars': 5,
        'num_vectors': 5,
        'num_tensors': 5,
        'scalar_size': 1,
        'vector_size': (128,),
        'tensor_size': (128, 128),
        'input_addresses':[5,6],
        'output_addresses':[7]
    }
    
    miner_type = "loss"  # Change this to "loss" or "simple" as needed
    best_genome = main(miner_type, config)
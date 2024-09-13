import torch
from tqdm import tqdm
from copy import deepcopy
import time

from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.unpacker import HierarchicalGenomeUnpacker

def main():
    # Define the target function (y = x/2 by default)
    def target_function(x):
        return x / 2

    # Generate some sample data
    x_data = torch.linspace(0, 10, 100)
    y_data = target_function(x_data)

    population_size = 50
    num_meta_levels = 1
    genome_length = 10
    tournament_size = 5
    generations = 1000
    generation_iters = 10

    num_scalars = 3
    num_vectors = 2
    num_tensors = 0
    scalar_size = 1
    vector_size = (100,)
    tensor_size = (1, 1)

    # Initialize components
    central_memory = CentralMemory(num_scalars, num_vectors, num_tensors,
                                   scalar_size, vector_size, tensor_size)
    function_decoder = FunctionDecoder()

    # Initialize AutoML-Zero
    automl = AutoMLZero(
        population_size=population_size,
        num_meta_levels=num_meta_levels,
        genome_length=genome_length,
        tournament_size=tournament_size,
        central_memory=central_memory,
        function_decoder=function_decoder
    )

    # Initialize population and evaluate initial fitness
    population = automl.hierarchical_genome.genomes[-1]
    evaluate_population(population, x_data, y_data)
    best_genome_all_time = deepcopy(max(population, key=lambda g: g.fitness))

    # Initialize counters for ops/second calculation
    total_ops = 0
    start_time = time.time()

    # Evolution loop
    for generation in tqdm(range(generations)):
        for _ in range(generation_iters):
            # Tournament selection
            parent = automl.tournament_selection(population)
            
            # Create and mutate offspring
            offspring = automl.mutate(parent, 0)

            # Evaluate the offspring
            evaluate_genome(offspring, x_data, y_data)

            # Add offspring to population and remove oldest member
            population.append(offspring)
            population.pop(0)

            # Increment the operation counter
            total_ops += 1

        # Check for perfect solution
        best_genome = max(population, key=lambda g: g.fitness)
        if best_genome.fitness > best_genome_all_time.fitness:
            best_genome_all_time = deepcopy(best_genome)

        # Calculate and print ops/second
        elapsed_time = time.time() - start_time
        ops_per_second = total_ops / elapsed_time

        print(f"Generation {generation}: Best fitness = {best_genome.fitness:.6f}, Ops/second = {ops_per_second:.2f}")

        if best_genome.fitness == 1.0:
            print("Perfect solution found!")
            break

    # Get the best evolved function
    best_genome_idx, best_genome = max(enumerate(population), key=lambda x: x[1].fitness) 
    print(f"Evolution complete. Best fitness: {best_genome.fitness:.6f}")
    print(f"Total operations: {total_ops}, Total time: {elapsed_time:.2f} seconds")
    print(f"Average ops/second: {total_ops / elapsed_time:.2f}")

    return best_genome, best_genome_idx, population, best_genome_all_time

def evaluate_population(population, x_data, y_data):
    for genome in population:
        evaluate_genome(genome, x_data, y_data)

def evaluate_genome(genome, x_data, y_data):
    try:
        evolved_function = genome.function()
        predicted = evolved_function(x_data)
        mse = torch.mean((predicted - y_data) ** 2)
        genome.fitness = 1 / (1 + mse)  # Convert MSE to a fitness score (higher is better)
    except:
        genome.fitness = 0

if __name__ == "__main__":
    best_genome, best_genome_idx, population, best_genome_all_time = main()
    unpacker = HierarchicalGenomeUnpacker()
    print("Best genome function:")
    print(unpacker.unpack_function_genome(best_genome_all_time))
    print("Best genome operations:")
    print(best_genome_all_time.gene)
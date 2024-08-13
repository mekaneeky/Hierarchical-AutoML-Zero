import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from memory import HierarchicalMemoryArrays
from function_decoder import FunctionDecoder
from fitness import FitnessEvaluator
from evolutionary_algorithm import LevelWiseEvolutionaryAlgorithm

def main():
    # Initialize parameters
    population_size = 100
    num_meta_levels = 2
    genome_length = 3
    tournament_size = 50
    mutation_probability = 1.0
    num_generations = [10, 50]
    meta_epochs = 100

    num_scalars = 5
    num_vectors = 5
    num_tensors = 5
    scalar_size = 1
    vector_size = (28,)
    tensor_size = (28, 28)

    # Initialize components
    hierarchical_memory = HierarchicalMemoryArrays(num_meta_levels, num_scalars, num_vectors, num_tensors,
                                                   scalar_size, vector_size, tensor_size)
    function_decoder = FunctionDecoder()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)

    # Get a sample for input_data and desired_output
    input_data_sample, target_sample = next(iter(train_loader))
    input_data = input_data_sample.squeeze().numpy()
    desired_outputs = target_sample

    # Initialize fitness evaluator and evolutionary algorithm
    fitness_evaluator = FitnessEvaluator(desired_outputs)
    ea = LevelWiseEvolutionaryAlgorithm(
        population_size, num_meta_levels, genome_length, tournament_size,
        mutation_probability, hierarchical_memory, function_decoder, input_data
    )

    # Run evolution
    best_genome = ea.evolve(fitness_evaluator, num_generations, meta_epochs)

    # Print results
    print("\nBest Hierarchical Genome:")
    for level, genome in enumerate(best_genome.genomes):
        print(f"Level {level}: {genome.gene}")

    final_output = best_genome.execute(input_data)
    final_fitness = fitness_evaluator.evaluate(best_genome, input_data)
    print(f"\nFinal Fitness: {final_fitness:.4f}")

if __name__ == "__main__":
    main()
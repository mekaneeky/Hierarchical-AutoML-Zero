import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from automl.memory import HierarchicalMemoryArrays
from automl.function_decoder import FunctionDecoder
from automl.fitness import FitnessEvaluator
from automl.evolutionary_algorithm import LevelWiseEvolutionaryAlgorithm, AutoMLZero
from automl.genome import FunctionGenome

class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=1000, shuffle=True)

    # Initialize AutoML Zero components
    memory = HierarchicalMemoryArrays(num_levels=1, num_scalars=10, num_vectors=5, num_tensors=2, scalar_size=1, vector_size=(28,), tensor_size=(28, 28))
    function_decoder = FunctionDecoder()
    
    # # Extend FunctionDecoder with activation function operations
    # function_decoder.decoding_map.update({
    #     31: lambda x: torch.sigmoid(x),
    #     32: lambda x: torch.tanh(x),
    #     33: lambda x: torch.relu(x),
    #     34: lambda x: torch.sin(x),
    #     35: lambda x: torch.exp(x),
    #     36: lambda x: x ** 2,
    #     37: lambda x: torch.log(torch.abs(x) + 1e-8),
    #     38: lambda x: torch.sqrt(torch.abs(x) + 1e-8),
    # })

    # Initialize evolutionary algorithm
    population_size = 100
    genome_length = 5  # Length of the activation function genome
    tournament_size = 5
    mutation_probability = 0.1
    num_generations = 50

    automl = AutoMLZero(
        population_size=population_size,
        num_meta_levels=1,
        genome_length=genome_length,
        tournament_size=tournament_size,
        mutation_probability=mutation_probability,
        hierarchical_memory=memory,
        function_decoder=function_decoder,
        input_data=torch.randn(1, 28*28)  # Dummy input for initialization
    )

    base_population = automl.hierarchical_genome.genomes[-1]
    ## TODO load basic population evaluation
    ## 

    for meta_epoch in range(4):
        for level in range(automl.num_meta_levels):
            print(f"Evolving Level {level}")
            population = automl.hierarchical_genome.genomes[level]
            #self.evaluate_level(level, population, fitness_evaluator)#FIXME make manual
            
            for generation in range(num_generations[level]):
                # Tournament selection
                parent = self.tournament_selection(population, fitness_evaluator)

                # Create and mutate offspring
                offspring = self.mutate(parent, level)

                # Add offspring to population and remove oldest member
                population.append(offspring)
                population.pop(0)

                # Mark offspring for re-evaluation
                self.fitness_cache[self.genome_to_key(offspring)] = (None, True)

                # Update the level's population in the hierarchical genome
                self.hierarchical_genome.genomes[level] = population

                # Calculate and print best fitness
                best_fitness = max(self.get_fitness(genome, fitness_evaluator) for genome in population)
                print(f"Meta-Epoch {meta_epoch + 1} Generation {generation + 1}, Level {level}: Best Fitness = {best_fitness:.4f}")

            # Update lower_level_population for the next level with the whole population
            if level < self.num_meta_levels - 1:
                for genome in self.hierarchical_genome.genomes[level + 1]:
                    genome.lower_level_population = population
                    # Invalidate cache for higher level genomes
                    self.fitness_cache[self.genome_to_key(genome)] = (None, True)


    # Evolution loop
    for generation in range(num_generations):
        for genome in ea.hierarchical_genome.genomes[0]:  # We're only using one level
            # Decode genome into a function
            evolved_activation = genome.function()
            
            # Create neural network with evolved activation
            model = EvolvableNN(input_size=28*28, hidden_size=128, output_size=10, evolved_activation=evolved_activation).to(device)

            # Train and evaluate model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            
            #for epoch in range(1):  # Train for 5 epochs
            try:
                train_model(model, train_loader, criterion, optimizer, device)
            except Exception as e:
                print(e)
                continue

            
            accuracy = evaluate_model(model, test_loader, device)
            print(accuracy)
            # Update evolutionary algorithm
            genome.fitness = accuracy

        # Select best genomes for next generation
        # ea.hierarchical_genome.genomes[0] = ea.select_best_genomes(ea.hierarchical_genome.genomes[0], FitnessEvaluator(0), num_best=population_size)

        # Print best accuracy for this generation
        #best_accuracy = max(ea.get_fitness(genome, FitnessEvaluator(0)) for genome in ea.hierarchical_genome.genomes[0])
        #print(f"Generation {generation + 1}/{num_generations}, Best Accuracy: {best_accuracy:.4f}")
        best_genome = max(generation, key=lambda g: g.fitness)
        print(f"Generation {generation.number}: Best accuracy = {best_genome.fitness:.4f}")


    # Get best evolved activation function
    #best_genome = max(ea.hierarchical_genome.genomes[0], key=lambda g: ea.get_fitness(g, FitnessEvaluator(0)))
    #best_activation = best_genome.function()

    print("Evolution complete. Best evolved activation function found.")

if __name__ == "__main__":
    main()
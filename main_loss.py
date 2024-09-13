import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import deepcopy

from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.unpacker import HierarchicalGenomeUnpacker
from automl.models import BaselineNN

def main():
    # Set up data
    train_loader, val_loader = load_data()

    population_size = 50
    num_meta_levels = 1
    genome_length = 20
    tournament_size = 20
    generations = 100
    generation_iters = 10

    num_scalars = 5
    num_vectors = 5
    num_tensors = 5
    scalar_size = 1
    vector_size = (128,)
    tensor_size = (128, 128)

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

    # Train baseline model with standard loss
    baseline_model = BaselineNN(input_size=28*28, hidden_size=128, output_size=10)
    baseline_loss = torch.nn.CrossEntropyLoss()
    train(baseline_model, train_loader, baseline_loss)
    baseline_accuracy = evaluate(baseline_model, val_loader)
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    # Initialize population and evaluate initial fitness
    population = automl.hierarchical_genome.genomes[-1]
    evaluate_population(population, train_loader, val_loader)
    best_genome_all_time = deepcopy(max(population, key=lambda g: g.fitness))

    # Evolution loop
    for generation in tqdm(range(generations)):
        for _ in tqdm(range(generation_iters)):
            # Tournament selection
            parent = automl.tournament_selection(population)
            
            # Create and mutate offspring
            offspring = automl.mutate(parent, 0)

            # Create a model with the evolved loss function
            model = BaselineNN(input_size=28*28, hidden_size=128, output_size=10)
            evolved_loss = offspring.function()

            # Train the model
            try:
                train(model, train_loader, evolved_loss)

                # Evaluate the model
                accuracy = evaluate(model, val_loader)

                # Update the offspring's fitness
                offspring.fitness = accuracy
                # Add offspring to population and remove oldest member
                population.append(offspring)
                population.pop(0)

            except:
                offspring.fitness = -9999

        # Log the best performance in this generation
        best_genome = deepcopy(max(population, key=lambda g: g.fitness))
        if best_genome.fitness > best_genome_all_time.fitness:
            best_genome_all_time = deepcopy(best_genome)

        print(f"Generation {generation}: Best accuracy = {best_genome.fitness:.4f}")

    # Get the best evolved loss function
    best_genome_idx, best_genome = max(enumerate(population), key=lambda x: x[1].fitness) 
    print(f"Evolution complete. Best accuracy: {best_genome.fitness:.4f}")

    return best_genome, best_genome_idx, population, best_genome_all_time

def evaluate_population(population, train_loader, val_loader):
    for genome in tqdm(population):
        try:
            model = BaselineNN(input_size=28*28, hidden_size=128, output_size=10)
            evolved_loss = genome.function()
            train(model, train_loader, evolved_loss)
            genome.fitness = evaluate(model, val_loader)
        except:
            del model
            genome.fitness = -9999

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    return train_loader, val_loader

def train(model, train_loader, loss_function):
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == 10:
            break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

if __name__ == "__main__":
    best_genome, best_genome_idx, population, best_genome_all_time = main()
    unpacker = HierarchicalGenomeUnpacker()
    print(unpacker.unpack_function_genome(best_genome_all_time))
    print(best_genome_all_time.gene)
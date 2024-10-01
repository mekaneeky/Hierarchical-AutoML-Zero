import argparse
import requests
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import deque
from copy import deepcopy

from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import HierarchicalMemoryArrays, CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.unpacker import HierarchicalGenomeUnpacker
from automl.models import EvolvableNN, BaselineNN
from automl.gene_io import export_gene_to_json, import_gene_from_json

def migrate_genes(best_gene, migration_server_url, function_decoder):
    # Submit best gene
    requests.post(f"{migration_server_url}/submit_gene", json=export_gene_to_json(gene=best_gene))
    
    # Get mixed genes from server
    response = requests.get(f"{migration_server_url}/get_mixed_genes")
    received_genes_data = response.json()
    
    return [import_gene_from_json(gene_data = gene_data, function_decoder = function_decoder) for gene_data in received_genes_data]

def main(migration_server_url):

    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, transform=transform)
    

    population_size = 100
    num_meta_levels = 1
    genome_length = 5
    tournament_size = 70
    mutation_probability = 1.0
    generations = 1000
    generation_iters = 100


    num_scalars = 5
    num_vectors = 5
    num_tensors = 5
    scalar_size = 1
    vector_size = (128,)
    tensor_size = (128, 128)

    # Initialize components
    central_memory = CentralMemory( num_scalars, num_vectors, num_tensors,
                                    scalar_size, vector_size, tensor_size)
    function_decoder = FunctionDecoder()

    # Initialize AutoML-Zero
    automl = AutoMLZero(
        population_size=population_size,
        num_meta_levels = num_meta_levels,
        genome_length=genome_length,
        tournament_size=tournament_size,
        central_memory = central_memory,
        function_decoder = function_decoder
    )

    baseline_model = BaselineNN(input_size=28*28, hidden_size=128, output_size=10)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    train(baseline_model, train_loader)
    baseline_accuracy = evaluate(baseline_model, val_loader)
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    # Initialize population and evaluate initial fitness
    population = automl.hierarchical_genome.genomes[-1]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    evaluate_population(population, train_loader, val_loader)
    best_genome_all_time = deepcopy(max(population, key=lambda g: g.fitness))

    for generation in tqdm(range(generations)):
        for _ in tqdm(range(generation_iters)):
            
            parent = automl.tournament_selection(population)
            
            # Create and mutate offspring
            offspring = automl.mutate(parent, 0)
            fingerprint, cached_fitness = automl.functional_equivalence_check(offspring)
            if cached_fitness is not None:
                offspring.fitness = cached_fitness
                #if cached_fitness != -9999:
                #    population.append(offspring)
                #    population.pop(0)
                continue
            # Create a model with the evolved activation function
            model = EvolvableNN(
                input_size=28*28, 
                hidden_size=128, 
                output_size=10, 
                evolved_activation=offspring.function()
            )
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

            #breakpoint()
            # Train the model
            try:
                train(model, train_loader)

                # Evaluate the model
                accuracy = evaluate(model, val_loader)

                # Update the offspring's fitness
                offspring.fitness = accuracy
                automl.fec_cache[fingerprint] = accuracy
                # Add offspring to population and remove oldest member
                population.append(offspring)
                population.pop(0)

            except Exception as e:
               offspring.fitness = -9999
               automl.fec_cache[fingerprint] = -9999 

        # Log the best performance in this generation
        best_genome = deepcopy(max(population, key=lambda g: g.fitness))
        if best_genome.fitness > best_genome_all_time.fitness:
            best_genome_all_time = deepcopy(best_genome)
            export_gene_to_json(best_genome_all_time, "best_gene.json")
            #push_to_huggingface(repo_name, "best_gene.json", f"Best gene (Gen {generation}, Acc {accuracy:.4f})")
            
        print(f"Generation {generation}: Best accuracy = {best_genome.fitness:.4f}")
        print(f"Total unique algorithms seen: {len(automl.fec_cache.keys())}")

        # Migration step
        if generation % 20 == 0:  # Migrate every 5 generations
            best_gene = max(population, key=lambda g: g.fitness)
            received_genes = migrate_genes(best_gene, migration_server_url, function_decoder)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
            evaluate_population(received_genes, train_loader, val_loader)
            
            # Integrate received genes into population
            for received_gene in received_genes:
                if received_gene.fitness > min(population, key=lambda g: g.fitness).fitness:
                    population.pop(0)  # Remove worst gene
                    population.append(received_gene)

    # Get the best evolved activation function
    best_genome_idx, best_genome = max(enumerate(population), key=lambda x: x[1].fitness) 
    print(f"Evolution complete. Best accuracy: {best_genome_all_time.fitness:.4f}")

    # You can now use best_genome to create a model with the best evolved activation function
    best_model = EvolvableNN(
        input_size=28*28, 
        hidden_size=128, 
        output_size=10, 
        evolved_activation=best_genome_all_time.function()
    )

    return best_model, best_genome, best_genome_idx, population, best_genome_all_time

def evaluate_population(population, train_loader, val_loader):
    for genome in tqdm(population):
        try:
            model = EvolvableNN(
                input_size=28*28, 
                hidden_size=128, 
                output_size=10, 
                evolved_activation=genome.function()
            )
            train(model, train_loader)
            genome.fitness = evaluate(model, val_loader)
        except:
            del model
            genome.fitness = -9999



def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == 1:
            break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    del optimizer, criterion
        

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            if idx > 10:
                return correct/total
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--migration_server_url", required=True, help="URL of the migration server")
    args = parser.parse_args()
    
    main(args.migration_server_url)
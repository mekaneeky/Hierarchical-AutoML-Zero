import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import deepcopy

from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.models import EvolvableNN, BaselineNN
from automl.gene_io import export_gene_to_json

from config import Config  # Assume we have a config file

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == Config.TRAIN_BATCHES:
            break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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

def create_and_evaluate_model(genome, train_loader, val_loader):
    model = EvolvableNN(
        input_size=Config.INPUT_SIZE, 
        hidden_size=Config.HIDDEN_SIZE, 
        output_size=Config.OUTPUT_SIZE, 
        evolved_activation=genome.function()
    )
    train(model, train_loader)
    return evaluate(model, val_loader)

def mine_genes(repo_name):
    train_loader, val_loader = load_data()

    central_memory = CentralMemory(
        Config.NUM_SCALARS, Config.NUM_VECTORS, Config.NUM_TENSORS,
        Config.SCALAR_SIZE, Config.VECTOR_SIZE, Config.TENSOR_SIZE
    )
    function_decoder = FunctionDecoder()

    automl = AutoMLZero(
        population_size=Config.POPULATION_SIZE,
        num_meta_levels=Config.NUM_META_LEVELS,
        genome_length=Config.GENOME_LENGTH,
        tournament_size=Config.TOURNAMENT_SIZE,
        central_memory=central_memory,
        function_decoder=function_decoder
    )

    baseline_model = BaselineNN(input_size=Config.INPUT_SIZE, hidden_size=Config.HIDDEN_SIZE, output_size=Config.OUTPUT_SIZE)
    train(baseline_model, train_loader)
    baseline_accuracy = evaluate(baseline_model, val_loader)
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    population = automl.hierarchical_genome.genomes[-1]
    best_genome_all_time = None
    best_fitness_all_time = -float('inf')

    for generation in tqdm(range(Config.NUM_GENERATIONS)):
        for _ in range(Config.GENERATION_ITERS):
            parent = automl.tournament_selection(population)
            offspring = automl.mutate(parent, 0)

            try:
                accuracy = create_and_evaluate_model(offspring, train_loader, val_loader)
                offspring.fitness = accuracy
                population.append(offspring)
                population.pop(0)

                if accuracy > best_fitness_all_time:
                    best_fitness_all_time = accuracy
                    best_genome_all_time = deepcopy(offspring)
                    
                    export_gene_to_json(best_genome_all_time, "best_gene.json")
                    push_to_huggingface(repo_name, "best_gene.json", f"Best gene (Gen {generation}, Acc {accuracy:.4f})")

            except Exception as e:
                print(f"Error in generation {generation}: {str(e)}")
                offspring.fitness = Config.MIN_FITNESS

        print(f"Generation {generation}: Best accuracy = {best_fitness_all_time:.4f}")

def push_to_huggingface(repo_name, file_path, commit_message):
    from huggingface_hub import HfApi, Repository
    import os

    api = HfApi()
    repo_url = f"https://huggingface.co/{repo_name}"
    
    if not os.path.exists(repo_name):
        Repository(repo_name, clone_from=repo_url)
    
    repo = Repository(repo_name, repo_url)
    repo.git_pull()
    
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_name,
        commit_message=commit_message
    )

if __name__ == "__main__":
    repo_name = "your-username/automl-mined-genes"  # Replace with your Hugging Face repo name
    mine_genes(repo_name)
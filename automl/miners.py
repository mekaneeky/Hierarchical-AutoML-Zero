from abc import ABC, abstractmethod
from copy import deepcopy
from huggingface_hub import HfApi, Repository
import os
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging 
import pandas as pd
import time

from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.models import BaselineNN, EvolvableNN
from automl.gene_io import export_gene_to_json, import_gene_from_json
from automl.destinations import PushMixin, PoolPushDestination, HuggingFacePushDestination

class BaseMiner(ABC, PushMixin):
    def __init__(self, config):
        self.config = config
        self.central_memory = CentralMemory(
            config.Miner.num_scalars, config.Miner.num_vectors, config.Miner.num_tensors,
            config.Miner.scalar_size, config.Miner.vector_size, config.Miner.tensor_size
        )
        self.function_decoder = FunctionDecoder()
        self.automl = AutoMLZero(
            population_size=config.Miner.population_size,
            num_meta_levels=config.Miner.num_meta_levels,
            genome_length=config.Miner.genome_length,
            tournament_size=config.Miner.tournament_size,
            central_memory=self.central_memory,
            function_decoder=self.function_decoder,
            input_addresses=config.Miner.input_addresses,
            output_addresses=config.Miner.output_addresses
        )
    
        self.migration_server_url = config.Miner.migration_server_url
        self.migration_interval = config.Miner.migration_interval
        self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []
        
        self.push_destinations = []

    def log_metrics(self, generation, accuracy):
        self.metrics_data.append({'generation': generation, 'accuracy': accuracy})
        
        # Save to CSV every 10 generations
        if len(self.metrics_data) % 10 == 0:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

    def get_best_migrant(self):
        response = requests.get(f"{self.migration_server_url}/get_best_fitness")
        best_fitness = response.json()["best_fitness"]
        if type(best_fitness) == float:
            return best_fitness
        else:
            return -1

    def emigrate_genes(self, best_gene):
        

        # Submit best gene
        gene_data = export_gene_to_json(gene=best_gene)
        response = requests.post(f"{self.migration_server_url}/submit_gene", json=gene_data)

        if response.status_code == 200:
            return True
        else:
            return False
        
    def immigrate_genes(self):
        # Get mixed genes from server
        response = requests.get(f"{self.migration_server_url}/get_mixed_genes")
        received_genes_data = response.json()

        if not self.migration_server_url:
            return []
        
        return [import_gene_from_json(gene_data=gene_data, function_decoder=self.function_decoder) 
                for gene_data in received_genes_data]

    #One migration cycle
    def migrate_genes(self,best_gene):
        self.emigrate_genes(best_gene)
        return self.immigrate_genes()
    
    def push_to_huggingface(self, file_path, commit_message):
        if not self.config.gene_repo:
            logging.info("No repository name provided. Skipping push to Hugging Face.")
            return

        api = HfApi()
        repo_url = f"https://huggingface.co/{self.config.gene_repo}"
        
        if not os.path.exists(self.config.gene_repo):
            Repository(self.config.gene_repo, clone_from=repo_url)
        
        repo = Repository(self.config.gene_repo, repo_url)
        repo.git_pull()
        
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=self.config.gene_repo,
            commit_message=commit_message
        )

    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10)

    def measure_baseline(self):
        train_loader, val_loader = self.load_data()
        baseline_model = self.create_baseline_model()
        self.train(baseline_model, train_loader)
        self.baseline_accuracy = self.evaluate(baseline_model, val_loader)
        logging.info(f"Baseline model accuracy: {self.baseline_accuracy:.4f}")
    
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self, genome):
        pass

    @abstractmethod
    def train(self, model, train_loader):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    def mine(self):
        self.measure_baseline()
        train_loader, val_loader = self.load_data()
        population = self.automl.hierarchical_genome.genomes[-1]
        self.evaluate_population(population, train_loader, val_loader)
        best_genome_all_time = deepcopy(max(population, key=lambda g: g.fitness))

        for generation in tqdm(range(self.config.Miner.generations)):
            for _ in tqdm(range(self.config.Miner.generation_iters)):
                parent = self.automl.tournament_selection(population)
                offspring = self.automl.mutate(parent, 0)
                
                fingerprint, cached_fitness = self.automl.functional_equivalence_check(offspring)
                if cached_fitness is not None:
                    offspring.fitness = cached_fitness
                    continue

                model = self.create_model(offspring)
                try:
                    self.train(model, train_loader)
                    accuracy = self.evaluate(model, val_loader)
                    offspring.fitness = accuracy
                    self.automl.fec_cache[fingerprint] = accuracy
                    population.append(offspring)
                    population.pop(0)
                    
                except Exception as e:
                    offspring.fitness = -9999
                    self.automl.fec_cache[fingerprint] = -9999

            best_genome = deepcopy(max(population, key=lambda g: g.fitness))
            self.log_metrics(generation, best_genome.fitness)
            
            if best_genome.fitness > best_genome_all_time.fitness:
                best_genome_all_time = deepcopy(best_genome)
                logging.info(f"New best gene found. Pushing to {self.config.gene_repo}")
                if self.migration_server_url:
                    best_migration_fitness = self.get_best_migrant()
                    if best_migration_fitness < best_genome_all_time.fitness:
                        self.push_to_remote(best_genome_all_time, f"Best gene (Gen {generation}, Acc {best_genome_all_time.fitness:.4f})")
                        self.emigrate_genes(best_genome_all_time)
                else:
                    self.push_to_remote(best_genome_all_time, f"Best gene (Gen {generation}, Acc {best_genome_all_time.fitness:.4f})")
                                
            logging.info(f"Generation {generation}: Best accuracy = {best_genome.fitness:.4f}")
            logging.info(f"Improvement over baseline: {best_genome.fitness - self.baseline_accuracy:.4f}")
            logging.info(f"Total unique algorithms seen: {len(self.automl.fec_cache.keys())}")

            if self.migration_server_url and generation % self.migration_interval == 0:
                received_genes = self.migrate_genes(best_genome)
                self.evaluate_population(received_genes, train_loader, val_loader)
                
                # Integrate received genes into population
                for received_gene in received_genes:
                    if received_gene.fitness > min(population, key=lambda g: g.fitness).fitness:
                        population.pop(0)  # Remove worst gene
                        population.append(received_gene)

        
        return best_genome_all_time
    
    def evaluate_population(self, population, train_loader, val_loader):
        for genome in tqdm(population):
            try:
                model = self.create_model(genome)
                self.train(model, train_loader)
                genome.fitness = self.evaluate(model, val_loader)
            except:
                genome.fitness = -9999

    @staticmethod
    def setup_logging(log_file='miner.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

class BaseHuggingFaceMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(HuggingFacePushDestination(config.gene_repo))

class BaseMiningPoolMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(PoolPushDestination(config.Miner.pool_url, config.bittensor_network.wallet))
        self.pool_url = config.Miner.pool_url

    # def mine_in_pool(self):
    #     # Register with the pool
    #     self.register_with_pool()

    #     # Get task from the pool
    #     task = self.get_task_from_pool()

    #     # Update config with pool task if necessary
    #     self.update_config_with_task(task)

    #     # Perform mining
    #     best_genome = self.mine()

    #     # Submit result to the pool
    #     self.submit_result_to_pool(best_genome)

    #     # Get rewards
    #     rewards = self.get_rewards_from_pool()

    #     return rewards

    #TODO add a timestamp or sth to requests to prevent spoofing signatures
    def register_with_pool(self):
        data = self._prepare_request_data("register")
        response = requests.post(f"{self.pool_url}/register", json=data)
        return response.json()['success']

    def get_task_from_pool(self):
        data = self._prepare_request_data("get_task")
        response = requests.get(f"{self.pool_url}/get_task", json=data)
        return response.json()

    def submit_result_to_pool(self, best_genome):
        data = self._prepare_request_data("submit_result")
        data["result"] = export_gene_to_json(best_genome)
        response = requests.post(f"{self.pool_url}/submit_result", json=data)
        return response.json()['success']

    def get_rewards_from_pool(self):
        data = self._prepare_request_data("get_rewards")
        response = requests.get(f"{self.pool_url}/get_rewards", json=data)
        return response.json() 

    def update_config_with_task(self, task):
        # Update miner config with task-specific parameters if needed
        pass

class ActivationMiner(BaseMiner):
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        return train_loader, val_loader

    def create_model(self, genome):
        return EvolvableNN(
            input_size=28*28, 
            hidden_size=128, 
            output_size=10, 
            evolved_activation=genome.function()
        )

    def train(self, model, train_loader):
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

    def evaluate(self, model, val_loader):
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
    
class EvolvedLoss(torch.nn.Module):
    def __init__(self, genome):
        super().__init__()
        self.genome = genome

    def forward(self, outputs, targets):
        outputs = outputs.detach().float().requires_grad_()
        targets = targets.detach().float().requires_grad_()
        
        memory = self.genome.memory
        memory.reset()
        
        memory[0] = outputs
        memory[1] = targets

        for i, op in enumerate(self.genome.gene):
            func = self.genome.function_decoder.decoding_map[op][0]
            input1 = memory[self.genome.input_gene[i]]
            input2 = memory[self.genome.input_gene_2[i]]
            constant = torch.tensor(self.genome.constants_gene[i], requires_grad=True)
            constant_2 = torch.tensor(self.genome.constants_gene_2[i], requires_grad=True)
            
            output = func(input1, input2, constant, constant_2, self.genome.row_fixed, self.genome.column_fixed)
            memory[self.genome.output_gene[i]] = output

        loss = memory[0].mean() if memory[0].numel() > 1 else memory[0]
        return loss

class LossMiner(BaseMiner):
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        return train_loader, val_loader

    def create_model(self, genome):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10), EvolvedLoss(genome)

    def train(self, model_and_loss, train_loader):
        model, loss_function = model_and_loss
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            if idx == 1:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate(self, model_and_loss, val_loader):
        model, _ = model_and_loss
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                if idx > 10:
                    break
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total
    
    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10), torch.nn.CrossEntropyLoss()
    
class SimpleMiner(BaseMiner):
    def load_data(self):
        x_data = torch.linspace(0, 10, 100)
        y_data = self.target_function(x_data)
        return (x_data, y_data), None

    def create_model(self, genome):
        return genome.function()

    def train(self, model, train_data):
        # No training needed for this simple case
        pass

    def evaluate(self, model, val_data):
        x_data, y_data = val_data
        try:
            predicted = model(x_data)
            mse = torch.mean((predicted[0] - y_data) ** 2)
            return 1 / (1 + mse)  # Convert MSE to a fitness score (higher is better)
        except:
            return 0

    @staticmethod
    def target_function(x):
        return (x / 2) + 2
    
class ActivationMinerPool(ActivationMiner, BaseMiningPoolMiner):
    pass

class ActivationMinerHF(ActivationMiner, BaseHuggingFaceMiner):
    pass

class LossMinerPool(LossMiner, BaseMiningPoolMiner):
    pass

class LossMinerHF(LossMiner, BaseHuggingFaceMiner):
    pass

class SimpleMinerPool(SimpleMiner, BaseMiningPoolMiner):
    pass

class SimpleMinerHF(SimpleMiner, BaseHuggingFaceMiner):
    pass


class MinerFactory:
    @staticmethod
    def get_miner(config):
        miner_type = config.Miner.miner_type
        platform = config.Miner.push_platform

        if platform == 'pool':
            if miner_type == "activation":
                return ActivationMinerPool(config)
            elif miner_type == "loss":
                return LossMinerPool(config)
        elif platform == 'hf':
            if miner_type == "activation":
                return ActivationMinerHF(config)
            elif miner_type == "loss":
                return LossMinerHF(config)
        
        raise ValueError(f"Unknown miner type: {miner_type} or platform: {platform}")
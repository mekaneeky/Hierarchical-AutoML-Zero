import heapq
import logging
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from abc import ABC, abstractmethod
import time
from huggingface_hub import HfApi, Repository

from automl.models import EvolvableNN, BaselineNN
from automl.gene_io import import_gene_from_json
from automl.function_decoder import FunctionDecoder

class BaseValidator(ABC):
    def __init__(self, config):
        self.config = config
        self.function_decoder = FunctionDecoder()
        self.device = config.device
        #self.hf_repo = config.hf_repo
        self.chain_manager = config.chain_manager
        self.bittensor_network = config.bittensor_network
        self.interval = config.Validator.validation_interval
        self.scores = {}
        self.normalized_scores = {}
        #self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []

    def log_metrics(self, iteration, scores):
        self.metrics_data.append({'iteration': iteration, **scores})
        
        # Save to CSV every 5 iterations
        if len(self.metrics_data) % 5 == 0:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self, gene):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10)

    def measure_baseline(self):
        _, val_loader = self.load_data()
        baseline_model = self.create_baseline_model()
        baseline_model.to(self.device)
        self.base_accuracy = self.evaluate(baseline_model, val_loader)
        logging.info(f"Baseline model accuracy: {self.base_accuracy:.4f}")

    def validate_and_score(self):
        logging.info("Receiving genes from chain")
        self.bittensor_network.sync(lite=True)
        
        if not self.check_registration():
            logging.info("This validator is no longer registered on the chain.")
            return

        _, val_loader = self.load_data()
        total_scores = 0


        for uid, hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
            hf_repo = self.chain_manager.retrieve_hf_repo(hotkey_address)
            gene = self.receive_gene_from_hf(hf_repo)

            if gene is not None:
                logging.info(f"Receiving gene from: {hotkey_address} ---> {hf_repo}")

                model = self.create_model(gene)
                if type(model) == tuple:
                    model[0].to(self.device)
                else:
                    model.to(self.device)

                accuracy = self.evaluate(model, val_loader)
                accuracy_score = max(0, accuracy - self.base_accuracy)
                total_scores += accuracy_score
                
                self.scores[hotkey_address] = accuracy_score
                logging.info(f"Accuracy: {accuracy:.4f}")
                logging.info(f"Accuracy Score: {accuracy_score:.4f}")
            else:
                logging.info(f"No gene received from: {hotkey_address}")
                self.scores[hotkey_address] = 0

        #self.log_metrics(len(self.metrics_data), self.scores)
        
        top_k = self.config.Validator.top_k
        min_score = self.config.Validator.min_score

        # Create a list of (score, hotkey) tuples
        score_hotkey_pairs = [(score, hotkey) for hotkey, score in self.scores.items()]

        # Get the top-k scores and their corresponding hotkeys
        top_k_scores = heapq.nlargest(top_k, score_hotkey_pairs)

        # Calculate the total score of the top-k miners
        total_top_k_score = sum(score for score, _ in top_k_scores)

        for hotkey_address in self.bittensor_network.metagraph.hotkeys:
            if hotkey_address in [hotkey for _, hotkey in top_k_scores]:
                # Normalize the score for top-k miners
                original_score = self.scores[hotkey_address]
                self.normalized_scores[hotkey_address] = original_score / total_top_k_score if total_top_k_score > 0 else 0
            else:
                # Assign min_score or 0 to miners not in the top-k
                self.normalized_scores[hotkey_address] = min_score
        logging.info(f"Normalized scores: {self.normalized_scores}")
        
        self.log_metrics(len(self.metrics_data), self.normalized_scores)

        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.normalized_scores)
            logging.info("Weights Setting attempted !")


    def check_registration(self):
        try:
            return self.bittensor_network.subtensor.is_hotkey_registered(
                netuid=self.bittensor_network.metagraph.netuid,
                hotkey_ss58=self.bittensor_network.wallet.hotkey.ss58_address
            )
        except:
            logging.warning("Failed to check registration, assuming still registered")
            return True

    def receive_gene_from_hf(self, repo_name):
        api = HfApi()
        try:
            file_info = api.list_repo_files(repo_id=repo_name)
            if "best_gene.json" in file_info:
                file_details = api.list_repo_tree(repo_id=repo_name, path="best_gene.json")
                if file_details:
                    file_size = file_details[0].size  # Size in bytes
                    max_size = self.config.max_gene_size  # 1 MB limit (adjust as needed) 1024*1024
                    
                    if file_size > max_size:
                        logging.warning(f"Gene file size ({file_size} bytes) exceeds limit ({max_size} bytes). Skipping download.")
                        return None
                    
                    gene_path = api.hf_hub_download(repo_id=repo_name, filename="best_gene.json")
                    gene_content = import_gene_from_json(filename=gene_path, config=self.config)
                    os.remove(gene_path)
                    return gene_content
                else:
                    logging.warning("Could not retrieve file details for best_gene.json")
            else:
                logging.info("best_gene.json not found in the repository")
        except Exception as e:
            logging.info(f"Error retrieving gene from Hugging Face: {str(e)}")
        return None

    def start_periodic_validation(self):
        while True:
            self.validate_and_score()
            logging.info(f"One round done, sleeping for: {self.interval}")
            time.sleep(self.interval)

    # @staticmethod
    # def setup_logging(log_file='validator.log'):
    #     logging.basicConfig(
    #         filename=log_file,
    #         level=logging.INFO,
    #         format='%(asctime)s - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S'
    #     )


class ActivationValidator(BaseValidator):
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        return train_loader, val_loader

    def create_model(self, gene):
        return EvolvableNN(
            input_size=28*28, 
            hidden_size=128, 
            output_size=10, 
            evolved_activation=gene.function()
        )

    def evaluate(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

class EvolvedLoss(torch.nn.Module):
    def __init__(self, gene):
        super().__init__()
        self.gene = gene

    def forward(self, outputs, targets):
        outputs = outputs.detach().float().requires_grad_()
        targets = targets.detach().float().requires_grad_()
        
        memory = self.gene.memory
        memory.reset()
        
        memory[0] = outputs
        memory[1] = targets

        for i, op in enumerate(self.gene.gene):
            func = self.gene.function_decoder.decoding_map[op][0]
            input1 = memory[self.gene.input_gene[i]]
            input2 = memory[self.gene.input_gene_2[i]]
            constant = torch.tensor(self.gene.constants_gene[i], requires_grad=True)
            constant_2 = torch.tensor(self.gene.constants_gene_2[i], requires_grad=True)
            
            output = func(input1, input2, constant, constant_2, self.gene.row_fixed, self.gene.column_fixed)
            memory[self.gene.output_gene[i]] = output

        loss = memory[0].mean() if memory[0].numel() > 1 else memory[0]
        return loss

class LossValidator(BaseValidator):
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        return train_loader, val_loader

    def create_model(self, gene):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10), EvolvedLoss(gene)

    def evaluate(self, model_and_loss, val_loader):
        model, loss_function = model_and_loss
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10), torch.nn.CrossEntropyLoss()
    
    def measure_baseline(self):
        _, val_loader = self.load_data()
        baseline_model, loss = self.create_baseline_model()
        baseline_model.to(self.device)
        self.base_accuracy = self.evaluate((baseline_model, loss), val_loader)
        logging.info(f"Baseline model accuracy: {self.base_accuracy:.4f}")

class ValidatorFactory:
    @staticmethod
    def get_validator(config):
        validator_type = config.Validator.validator_type
        if validator_type == "activation":
            return ActivationValidator(config)
        elif validator_type == "loss":
            return LossValidator(config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")
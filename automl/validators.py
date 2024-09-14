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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo = config['hf_repo']
        self.chain_manager = config['chain_manager']
        self.bittensor_network = config['bittensor_network']
        self.interval = config['validation_interval']
        self.scores = {}
        self.normalized_scores = {}

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
        print(f"Baseline model accuracy: {self.base_accuracy:.4f}")

    def validate_and_score(self):
        print("Receiving genes from chain")
        self.bittensor_network.sync(lite=True)

        if not self.check_registration():
            print("This validator is no longer registered on the chain.")
            return

        _, val_loader = self.load_data()
        total_scores = 0

        for uid, hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
            hf_repo = self.chain_manager.retrieve_hf_repo(hotkey_address)
            gene = self.receive_gene_from_hf(hf_repo)
            
            if gene is not None:
                print(f"Receiving gene from: {hotkey_address}")
                imported_gene = import_gene_from_json(gene, self.function_decoder)
                
                model = self.create_model(imported_gene)
                model.to(self.device)
                
                accuracy = self.evaluate(model, val_loader)
                accuracy_score = max(0, accuracy - self.base_accuracy)
                total_scores += accuracy_score
                
                self.scores[hotkey_address] = accuracy_score
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Accuracy Score: {accuracy_score:.4f}")
            else:
                print(f"No gene received from: {hotkey_address}")
                self.scores[hotkey_address] = 0

        # Normalize scores
        for hotkey_address in self.bittensor_network.metagraph.hotkeys:
            self.normalized_scores[hotkey_address] = max(0, self.scores[hotkey_address] / total_scores) if total_scores > 0 else 0

        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.normalized_scores)

    def check_registration(self):
        return self.bittensor_network.subtensor.is_hotkey_registered(
            netuid=self.bittensor_network.metagraph.netuid,
            hotkey_ss58=self.bittensor_network.wallet.hotkey.ss58_address
        )

    def receive_gene_from_hf(self, repo_name):
        api = HfApi()
        try:
            file_info = api.list_repo_files(repo_id=repo_name)
            if "best_gene.json" in file_info:
                gene_content = api.hf_hub_download(repo_id=repo_name, filename="best_gene.json")
                return gene_content
        except Exception as e:
            print(f"Error retrieving gene from Hugging Face: {str(e)}")
        return None

    def start_periodic_validation(self):
        while True:
            self.validate_and_score()
            print(f"One round done, sleeping for: {self.interval}")
            time.sleep(self.interval)

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
    
class ValidatorFactory:
    @staticmethod
    def get_validator(validator_type, config):
        if validator_type == "activation":
            return ActivationValidator(config)
        elif validator_type == "loss":
            return LossValidator(config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")
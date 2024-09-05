import torch
import torch.nn as nn
import torch.nn.functional as F
from automl.evolutionary_algorithm import AutoMLZero
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder
from automl.models import EvolvableNN
from automl.gene_io import import_gene_from_json
from bittensor import logging

class AutoMLValidator:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.central_memory = CentralMemory(num_scalars=5, num_vectors=5, num_tensors=5,
                                            scalar_size=1, vector_size=(128,), tensor_size=(128, 128))
        self.function_decoder = FunctionDecoder()

    def validate_and_score(self):
        logging.info("Receiving genes from chain")
        self.bittensor_network.sync(lite=True)

        if not self.check_registration():
            logging.warning("This validator is no longer registered on the chain.")
            return

        total_scores = 0
        for uid, hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
            hf_repo = self.chain_manager.retrieve_hf_repo(hotkey_address)
            gene = self.hf_manager.receive_gene(hf_repo)
            
            if gene is not None:
                logging.info(f"Receiving gene from: {hotkey_address}")
                imported_gene = import_gene_from_json(gene, self.function_decoder)
                
                model = EvolvableNN(
                    input_size=28*28,
                    hidden_size=128,
                    output_size=10,
                    evolved_activation=imported_gene.function()
                )
                
                loss, accuracy = self.evaluate_automl_model(model)
                accuracy_score = max(0, accuracy - self.base_accuracy)
                total_scores += accuracy_score
                
                self.scores[hotkey_address] = accuracy_score
                logging.info(f"Loss: {loss}, Accuracy: {accuracy}")
                logging.info(f"Accuracy Score: {accuracy_score}")
            else:
                logging.warning(f"No gene received from: {hotkey_address}")
                self.scores[hotkey_address] = 0

        # Normalize scores
        for hotkey_address in self.bittensor_network.metagraph.hotkeys:
            self.normalized_scores[hotkey_address] = max(0, self.scores[hotkey_address] / total_scores) if total_scores > 0 else 0

        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.normalized_scores)

    def evaluate_automl_model(self, model):
        model.to(self.device)
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

    def check_registration(self):
        return self.bittensor_network.subtensor.is_hotkey_registered(
            netuid=self.bittensor_network.metagraph.netuid,
            hotkey_ss58=self.bittensor_network.wallet.hotkey.ss58_address
        )

    def start_periodic_validation(self):
        while True:
            self.validate_and_score()
            self.hf_manager.clear_hf_cache()
            logging.info(f"One round done sleeping for: {self.interval}")
            time.sleep(self.interval)
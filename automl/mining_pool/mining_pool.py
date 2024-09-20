from flask import Flask, request, jsonify
from miners import MinerFactory
from auth import authenticate_request_with_bittensor
from automl.gene_io import import_gene_from_json, export_gene_to_json
import torch
import logging

app = Flask(__name__)

class MiningPool:
    def __init__(self, config):
        self.config = config
        self.miners = {}
        self.results = {}
        self.rewards = {}
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def register_miner(self, miner_address):
        if miner_address not in self.miners:
            self.miners[miner_address] = MinerFactory.get_miner(self.config)
            logging.info(f"Miner {miner_address} registered")
        return True

    def distribute_task(self):
        # In this simple implementation, all miners work on the same task
        return self.config.Miner

    def collect_result(self, miner_address, result):
        self.results[miner_address] = result
        logging.info(f"Received result from miner {miner_address}")

    def evaluate_results(self):
        best_fitness = float('-inf')
        best_miner = None
        for miner_address, result in self.results.items():
            fitness = result['fitness']
            if fitness > best_fitness:
                best_fitness = fitness
                best_miner = miner_address
        return best_miner, best_fitness

    def calculate_rewards(self):
        best_miner, best_fitness = self.evaluate_results()
        total_reward = 100  # Placeholder value
        for miner_address in self.results:
            if miner_address == best_miner:
                self.rewards[miner_address] = total_reward * 0.5  # 50% to the best miner
            else:
                self.rewards[miner_address] = total_reward * 0.5 / (len(self.results) - 1)  # Split the rest

    def distribute_rewards(self):
        for miner_address, reward in self.rewards.items():
            # Here you would actually send the reward to the miner's wallet
            logging.info(f"Sending reward of {reward} to miner {miner_address}")

mining_pool = MiningPool(app.config)

@app.route('/register', methods=['POST'])
@authenticate_request_with_bittensor
def register():
    data = request.json
    miner_address = data['public_address']
    success = mining_pool.register_miner(miner_address)
    return jsonify({"success": success})

@app.route('/get_task', methods=['GET'])
@authenticate_request_with_bittensor
def get_task():
    task = mining_pool.distribute_task()
    return jsonify(task)

@app.route('/submit_result', methods=['POST'])
@authenticate_request_with_bittensor
def submit_result():
    data = request.json
    miner_address = data['public_address']
    result = import_gene_from_json(gene_data=data['result'])
    mining_pool.collect_result(miner_address, result)
    return jsonify({"success": True})

@app.route('/get_rewards', methods=['GET'])
@authenticate_request_with_bittensor
def get_rewards():
    mining_pool.calculate_rewards()
    mining_pool.distribute_rewards()
    return jsonify(mining_pool.rewards)

if __name__ == '__main__':
    app.run(port=5000)
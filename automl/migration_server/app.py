# migration_server.py
from flask import Flask, request, jsonify
import json
import random
from threading import Lock

app = Flask(__name__)

gene_pool = []
best_fitness = None
pool_lock = Lock()

@app.route('/submit_gene', methods=['POST'])
def submit_gene():
    gene = request.json
    global gene_pool, best_fitness
    
    with pool_lock:
        gene_pool.append(gene)
        gene_pool.sort(key=lambda x: x['fitness'], reverse=True)
        gene_pool = gene_pool[:100]  # Keep top 100 genes
        
        # Update best_fitness if necessary
        if best_fitness is None or gene['fitness'] > best_fitness:
            best_fitness = gene['fitness']
    
    return jsonify({"status": "success"})

@app.route('/get_mixed_genes', methods=['GET'])
def get_mixed_genes():
    with pool_lock:
        num_genes = min(10, len(gene_pool))  # Return up to 10 genes
        mixed_genes = random.sample(gene_pool, num_genes)
    return jsonify(mixed_genes)

@app.route('/get_best_fitness', methods=['GET'])
def get_best_fitness():
    with pool_lock:
        return jsonify({"best_fitness": best_fitness})

if __name__ == '__main__':
    app.run(port=4999)
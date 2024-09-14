# migration_server.py
from flask import Flask, request, jsonify
import json
import random

app = Flask(__name__)

gene_pool = []

@app.route('/submit_gene', methods=['POST'])
def submit_gene():
    gene = request.json
    global gene_pool
    gene_pool.append(gene)
    gene_pool.sort(key=lambda x: x['fitness'], reverse=True)
    gene_pool = gene_pool[:100]  # Keep top 100 genes
    return jsonify({"status": "success"})

@app.route('/get_mixed_genes', methods=['GET'])
def get_mixed_genes():
    num_genes = min(10, len(gene_pool))  # Return up to 10 genes
    mixed_genes = random.sample(gene_pool, num_genes)
    return jsonify(mixed_genes)

if __name__ == '__main__':
    app.run(port=5000)
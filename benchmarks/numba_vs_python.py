import numpy as np
from numba import njit, float64
import time

# Shared operations for both approaches
@njit
def add(x, y): return x + y
@njit
def subtract(x, y): return x - y
@njit
def multiply(x, y): return x * y
@njit
def safe_divide(x, y): return x / y if y != 0 else x
@njit
def sine(x, _): return np.sin(x)

# Hybrid Approach: Numba + Python class
@njit
def execute_genome_hybrid(gene, input_gene, input_gene_2, output_gene, memory, input_data):
    memory[-1] = input_data
    for i in range(len(gene)):
        op = gene[i]
        x, y = memory[input_gene[i]], memory[input_gene_2[i]]
        if op == 0:
            result = add(x, y)
        elif op == 1:
            result = subtract(x, y)
        elif op == 2:
            result = multiply(x, y)
        elif op == 3:
            result = safe_divide(x, y)
        elif op == 4:
            result = sine(x, y)
        else:
            result = x  # No-op
        memory[output_gene[i]] = result
    return memory[0]

class FunctionGenome:
    def __init__(self, length, memory_size):
        self.gene = np.random.randint(0, 6, size=length)  # 0-4 for operations, 5 for no-op
        self.input_gene = np.random.randint(0, memory_size, size=length)
        self.input_gene_2 = np.random.randint(0, memory_size, size=length)
        self.output_gene = np.random.randint(0, memory_size, size=length)
        self.memory_size = memory_size

    def function(self):
        def evolved_function(input_data, memory=None):
            if memory is None:
                memory = np.zeros(self.memory_size)
            return execute_genome_hybrid(self.gene, self.input_gene, self.input_gene_2, self.output_gene, memory, input_data)
        return evolved_function

# Pure Numba Approach
@njit
def create_random_genome(length, memory_size):
    return (np.random.randint(0, 6, size=length),  # 0-4 for operations, 5 for no-op
            np.random.randint(0, memory_size, size=length),
            np.random.randint(0, memory_size, size=length),
            np.random.randint(0, memory_size, size=length))

@njit
def execute_genome_pure(gene, input_gene, input_gene_2, output_gene, memory, input_data):
    memory[-1] = input_data
    for i in range(len(gene)):
        op = gene[i]
        x, y = memory[input_gene[i]], memory[input_gene_2[i]]
        if op == 0:
            result = add(x, y)
        elif op == 1:
            result = subtract(x, y)
        elif op == 2:
            result = multiply(x, y)
        elif op == 3:
            result = safe_divide(x, y)
        elif op == 4:
            result = sine(x, y)
        else:
            result = x  # No-op
        memory[output_gene[i]] = result
    return memory[0]

# Benchmark function
def run_benchmark(num_genomes, genome_length, memory_size, num_executions):
    # Hybrid Approach
    start_time = time.time()
    hybrid_genomes = [FunctionGenome(genome_length, memory_size) for _ in range(num_genomes)]
    hybrid_functions = [genome.function() for genome in hybrid_genomes]
    for _ in range(num_executions):
        for func in hybrid_functions:
            _ = func(np.random.random())
    hybrid_time = time.time() - start_time

    # Pure Numba Approach
    start_time = time.time()
    pure_genomes = [create_random_genome(genome_length, memory_size) for _ in range(num_genomes)]
    memory = np.zeros(memory_size)
    for _ in range(num_executions):
        for gene, input_gene, input_gene_2, output_gene in pure_genomes:
            _ = execute_genome_pure(gene, input_gene, input_gene_2, output_gene, memory, np.random.random())
    pure_time = time.time() - start_time

    print(f"Hybrid Approach Time: {hybrid_time:.4f} seconds")
    print(f"Pure Numba Approach Time: {pure_time:.4f} seconds")
    print(f"Speedup factor: {hybrid_time / pure_time:.2f}x")

    total_ops = num_genomes * genome_length * num_executions
    
    hybrid_ops_per_sec = total_ops / hybrid_time
    pure_ops_per_sec = total_ops / pure_time

    print(f"Hybrid Approach Ops/sec: {hybrid_ops_per_sec:.2f}")
    print(f"Pure Numba Approach Ops/sec: {pure_ops_per_sec:.2f}")

if __name__ == "__main__":
    print("Running benchmark...")
    run_benchmark(num_genomes=10000, genome_length=500, memory_size=100, num_executions=100)
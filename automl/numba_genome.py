#FIXME incomplete example 
import random
import torch

import numpy as np
from numba import njit, float64

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
@njit
def cosine(x, _): return np.cos(x)
@njit
def exponential(x, _): return np.exp(x)
@njit
def logarithm(x, _): return np.log(np.abs(x) + 1e-7)
@njit
def relu(x, _): return max(0, x)

operations = (add, subtract, multiply, safe_divide, sine, cosine, exponential, logarithm, relu)

@njit
def execute_genome(gene, input_gene, input_gene_2, output_gene, memory, input_data):
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
    def __init__(self, length, memory_size, lower_level_population=None):
        self.gene = np.random.randint(0, len(operations) + 1, size=length)
        self.input_gene = np.random.randint(0, memory_size, size=length)
        self.input_gene_2 = np.random.randint(0, memory_size, size=length)
        self.output_gene = np.random.randint(0, memory_size, size=length)
        self.memory_size = memory_size
        self.lower_level_population = lower_level_population
        self.fitness = None

    def function(self):
        def evolved_function(input_data, memory=None):
            if memory is None:
                memory = np.zeros(self.memory_size)
            return execute_genome(self.gene, self.input_gene, self.input_gene_2, self.output_gene, memory, input_data)
        return evolved_function
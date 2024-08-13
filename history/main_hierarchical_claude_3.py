import torch
import random
import string
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import copy

class MemoryArrays:
    def __init__(self, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
        self.scalar_memory = [torch.zeros((1,)) for _ in range(num_scalars)] if num_scalars > 0 else []
        self.vector_memory = [torch.zeros(vector_size) for _ in range(num_vectors)] if num_vectors > 0 else []
        self.tensor_memory = [torch.zeros(tensor_size) for _ in range(num_tensors)] if num_tensors > 0 else []
        
        self.memory_types = []
        if num_scalars > 0:
            self.memory_types.append('scalar')
        if num_vectors > 0:
            self.memory_types.append('vector')
        if num_tensors > 0:
            self.memory_types.append('tensor')

        self._total_len = sum(len(getattr(self, f"{t}_memory")) for t in self.memory_types)

    def store_scalar(self, index, data):
        if 'scalar' not in self.memory_types:
            raise ValueError("Scalar memory is not initialized.")
        if 0 <= index < len(self.scalar_memory):
            if isinstance(data, torch.Tensor):
                data = data.item() if data.numel() == 1 else data.mean().item()
            elif isinstance(data, np.ndarray):
                data = data.item() if data.size == 1 else data.mean().item()
            elif isinstance(data, (list, tuple)):
                data = data[0] if len(data) == 1 else sum(data) / len(data)
            self.scalar_memory[index] = torch.tensor([float(data)])
        else:
            raise IndexError(f"Scalar memory index {index} out of range.")

    def store_vector(self, index, data):
        if 'vector' not in self.memory_types:
            raise ValueError("Vector memory is not initialized.")
        if 0 <= index < len(self.vector_memory):
            target_size = self.vector_memory[index].shape[0]
            if isinstance(data, (int, float)):
                data = torch.full((target_size,), float(data))
            elif isinstance(data, list):
                data = torch.tensor(data, dtype=torch.float)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, torch.Tensor):
                data = data.float()
            
            if data.dim() == 0:
                data = data.unsqueeze(0)
            if data.dim() > 1:
                data = data.flatten()
            if data.shape[0] < target_size:
                repeat_times = (target_size + data.shape[0] - 1) // data.shape[0]
                data = data.repeat(repeat_times)[:target_size]
            elif data.shape[0] > target_size:
                data = data[:target_size]
            self.vector_memory[index] = data
        else:
            raise IndexError(f"Vector memory index {index} out of range.")

    def store_tensor(self, index, data):
        if 'tensor' not in self.memory_types:
            raise ValueError("Tensor memory is not initialized.")
        if 0 <= index < len(self.tensor_memory):
            target_shape = self.tensor_memory[index].shape
            if isinstance(data, (int, float)):
                data = torch.full(target_shape, float(data))
            elif isinstance(data, list):
                data = torch.tensor(data, dtype=torch.float)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, torch.Tensor):
                data = data.float()
            
            if data.dim() == 1:
                data = data.unsqueeze(0)
            if data.shape != target_shape:
                data = data.flatten().repeat((target_shape[0] * target_shape[1] + data.numel() - 1) // data.numel())
                data = data[:target_shape[0] * target_shape[1]].reshape(target_shape)
            self.tensor_memory[index] = data
        else:
            raise IndexError(f"Tensor memory index {index} out of range.")

    def __getitem__(self, index):
        
        if index < 0:
            index = self._total_len + index

        if index >= self._total_len:
            raise IndexError(f"Memory index {index} out of range.")

        for memory_type in self.memory_types:
            memory = getattr(self, f"{memory_type}_memory")
            if index < len(memory):
                return memory[index]
            index -= len(memory)

    def __setitem__(self, index, data):
        if index < 0:
            index = self._total_len + index

        if index >= self._total_len:
            raise IndexError(f"Memory index {index} out of range.")

        for memory_type in self.memory_types:
            memory = getattr(self, f"{memory_type}_memory")
            if index < len(memory):
                if memory_type == 'scalar':
                    self.store_scalar(index, data)
                elif memory_type == 'vector':
                    self.store_vector(index, data)
                else:  # tensor
                    self.store_tensor(index, data)
                return
            index -= len(memory)

    def __len__(self):
        return sum(len(getattr(self, f"{t}_memory")) for t in self.memory_types)

class HierarchicalMemoryArrays:
    def __init__(self, num_levels, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
        self.memory_levels = [
            MemoryArrays(num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size)
            for _ in range(num_levels)
        ]

    def __getitem__(self, level):
        return self.memory_levels[level]

class FunctionDecoder:
    def __init__(self):
        self.decoding_map = {
            0: self.identity,
            1: self.add_scalar,
            2: self.sub_scalar,
            3: self.multiply_scalar,
            4: self.divide_scalar,
            5: self.abs_scalar,
            6: self.reciprocal_scalar,
            7: self.sin_scalar,
            8: self.cos_scalar,
            9: self.tan_scalar,
            10: self.arcsin_scalar,
            11: self.arccos_scalar,
            12: self.arctan_scalar,
            13: self.exp_scalar,
            14: self.log_scalar,
            15: self.power_scalar,
            16: self.sqrt_scalar,
            17: self.max_scalar,
            18: self.min_scalar,
            19: self.mod_scalar,
            20: self.sign_scalar,
            21: self.floor_scalar,
            22: self.ceil_scalar,
            23: self.round_scalar,
            24: self.hypot_scalar,
            25: self.logistic_scalar,
        }

    @staticmethod
    def identity(*args):
        return args[0]

    @staticmethod
    def add_scalar(*args):
        return np.add(args[0], args[1])

    @staticmethod
    def sub_scalar(*args):
        return np.subtract(args[0], args[1])

    @staticmethod
    def multiply_scalar(*args):
        return np.multiply(args[0], args[1])

    @staticmethod
    def divide_scalar(*args):
        return np.divide(args[0], args[1] + 1e-10)

    @staticmethod
    def abs_scalar(*args):
        return np.abs(args[0])

    @staticmethod
    def reciprocal_scalar(*args):
        return np.where(np.abs(args[0]) > 1e-10, 1.0 / args[0], 0.0)

    @staticmethod
    def sin_scalar(*args):
        return np.sin(args[0])

    @staticmethod
    def cos_scalar(*args):
        return np.cos(args[0])

    @staticmethod
    def tan_scalar(*args):
        return np.tan(np.clip(args[0], -np.pi/2 + 1e-10, np.pi/2 - 1e-10))

    @staticmethod
    def arcsin_scalar(*args):
        return np.arcsin(args[0])

    @staticmethod
    def arccos_scalar(*args):
        return np.arccos(np.clip(args[0], -1.0, 1.0))

    @staticmethod
    def arctan_scalar(*args):
        return np.arctan(args[0])

    @staticmethod
    def exp_scalar(*args):
        return np.exp(args[0])

    @staticmethod
    def log_scalar(*args):
        return np.log(np.abs(args[0]) + 1e-10)  # Adding small value to avoid log(0)

    @staticmethod
    def power_scalar(*args):
        return np.power(args[0], args[1])

    @staticmethod
    def sqrt_scalar(*args):
        return np.sqrt(np.abs(args[0]))  # Using abs to avoid complex numbers

    @staticmethod
    def max_scalar(*args):
        return np.maximum(args[0], args[1])

    @staticmethod
    def min_scalar(*args):
        return np.minimum(args[0], args[1])

    @staticmethod
    def mod_scalar(*args):
        return np.where(args[1] != 0, np.mod(args[0], args[1]), 0.0)

    @staticmethod
    def sign_scalar(*args):
        return np.sign(args[0])

    @staticmethod
    def floor_scalar(*args):
        return np.floor(args[0])

    @staticmethod
    def ceil_scalar(*args):
        return np.ceil(args[0])

    @staticmethod
    def round_scalar(*args):
        return np.round(args[0])

    @staticmethod
    def hypot_scalar(*args):
        return np.hypot(args[0], args[1])

    @staticmethod
    def logistic_scalar(*args):
        return 1 / (1 + np.exp(-args[0]))

    def decode(self, genome):
        decoded_functions = []
        for op in genome.gene:
            if op in self.decoding_map:
                decoded_functions.append(self.decoding_map[op])
        return decoded_functions, genome


class FunctionGenome:
    def __init__(self, length, hierarchical_memory, function_decoder, meta_level=0, lower_level_population=None):
        self.length = length
        self.hierarchical_memory = hierarchical_memory
        self.function_decoder = function_decoder
        self.meta_level = meta_level
        self.lower_level_population = lower_level_population

        # Calculate size of the lower level population
        self.lower_level_size = len(lower_level_population) if lower_level_population else 0

        max_op = max(function_decoder.decoding_map.keys())
        self.max_op_pop = max_op + self.lower_level_size
        self.gene = [random.randint(0, self.max_op_pop) for _ in range(length)]
        
        self.memory_size = len(hierarchical_memory[meta_level])
        self.input_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.input_gene_2 = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.output_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.version = 0

    def execute(self, input_data):
        memory = self.hierarchical_memory[self.meta_level]
        memory[-1] = input_data

        for i, op in enumerate(self.gene):
            if op in self.function_decoder.decoding_map:
                func = self.function_decoder.decoding_map[op]
                input1 = memory[self.input_gene[i]]
                input2 = memory[self.input_gene_2[i]]
                output = func(input1, input2)
            elif self.lower_level_population and op - len(self.function_decoder.decoding_map) < self.lower_level_size:
                lower_genome = self.lower_level_population[op - len(self.function_decoder.decoding_map)]
                output = lower_genome.execute(input_data)
            else:
                output = input_data  # No operation if op is out of range

            memory[self.output_gene[i]] = output

        return memory[0]  # Assuming the final output is always stored in the first memory location

    # The execute_hierarchical method is no longer needed as execute now handles both cases

    def mutate(self):
        mutation_type = random.choice(['one_argument', 'add_or_remove_one', 'all'])
        
        if mutation_type == 'one_argument':
            self.mutate_one_argument()
        elif mutation_type == 'add_or_remove_one':
            self.mutate_add_or_remove_one_instruction()
        else:  # mutation_type == 'all'
            self.mutate_all()
        self.version += 1

    def mutate_one_argument(self):
        instruction_idx = random.randint(0, self.length - 1)
        if random.random() > 0.5:
            # Mutate input argument
            gene_to_mutate = random.choice(['input_gene', 'input_gene_2'])
            gene_list = getattr(self, gene_to_mutate)
            gene_list[instruction_idx] = random.randint(0, self.memory_size - 1)
            setattr(self, gene_to_mutate, gene_list)
        else:
            # Mutate output argument
            self.output_gene[instruction_idx] = random.randint(0, self.memory_size - 1)

    def mutate_add_or_remove_one_instruction(self):
        instruction_idx = random.randint(0, self.length - 1)

        self.gene[instruction_idx] = random.randint(0, self.max_op_pop)

        for gene_name in ['input_gene', 'input_gene_2', 'output_gene']:
            gene_list = getattr(self, gene_name)
            gene_list[instruction_idx] = random.randint(0, self.memory_size - 1)
            setattr(self, gene_name, gene_list)

    def mutate_all(self):
        self.gene = [random.randint(0, self.max_op_pop) for _ in range(self.length)]
        self.input_gene = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]
        self.input_gene_2 = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]
        self.output_gene = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]


class HierarchicalFunctionGenome:
    def __init__(self, length, hierarchical_memory, meta_level, lower_level_population):
        self.length = length
        self.hierarchical_memory = hierarchical_memory
        self.meta_level = meta_level
        self.lower_level_population = lower_level_population

        self.memory_size = len(hierarchical_memory[meta_level])
        
        # Initialize genes
        self.gene = [random.randint(0, len(lower_level_population) - 1) for _ in range(length)]
        self.input_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.input_gene_2 = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.output_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]

    def mutate(self):
        mutation_type = random.choice(['function', 'input', 'output'])
        
        if mutation_type == 'function':
            index = random.randint(0, self.length - 1)
            self.gene[index] = random.randint(0, len(self.lower_level_population) - 1)
        elif mutation_type == 'input':
            index = random.randint(0, self.length - 1)
            input_to_mutate = random.choice(['input_gene', 'input_gene_2'])
            getattr(self, input_to_mutate)[index] = random.randint(0, self.memory_size - 1)
        else:  # output
            index = random.randint(0, self.length - 1)
            self.output_gene[index] = random.randint(0, self.memory_size - 1)

    def execute(self, input_data):
        memory = self.hierarchical_memory[self.meta_level]
        memory[-1] = input_data

        for i, function_index in enumerate(self.gene):
            lower_level_function = self.lower_level_population[function_index]
            
            input1 = memory[self.input_gene[i]]
            input2 = memory[self.input_gene_2[i]]
            
            # Execute the lower level function
            output = lower_level_function.execute(input1)
            
            # Store the result
            memory[self.output_gene[i]] = output

        return memory[0]  # Assuming the final output is always stored in the first memory location

    def __deepcopy__(self, memo):
        # Create a new instance
        new_copy = HierarchicalFunctionGenome(self.length, self.hierarchical_memory, self.meta_level, self.lower_level_population)
        
        # Copy the genes
        new_copy.gene = self.gene.copy()
        new_copy.input_gene = self.input_gene.copy()
        new_copy.input_gene_2 = self.input_gene_2.copy()
        new_copy.output_gene = self.output_gene.copy()
        
        return new_copy

class HierarchicalGenome:
    def __init__(self, num_meta_levels, genome_length, hierarchical_memory, function_decoder):
        self.num_meta_levels = num_meta_levels
        self.hierarchical_memory = hierarchical_memory
        self.function_decoder = function_decoder
        
        self.genomes = [
            [FunctionGenome(genome_length, hierarchical_memory, function_decoder, meta_level=i)
             for _ in range(population_size)] for i in range(num_meta_levels)
        ]


class FitnessEvaluator:
    def __init__(self, desired_outputs):
        self.desired_outputs = desired_outputs

    def evaluate(self, genome, input_data):
        output = genome.execute(input_data)
        if isinstance(output, torch.Tensor):
            output = output.mean().item()  # Convert tensor to scalar
        if isinstance(self.desired_outputs, torch.Tensor):
            desired_output = self.desired_outputs.mean().item()
        else:
            desired_output = self.desired_outputs
        return 1.0 / (1.0 + abs(output - desired_output))
    
class LevelWiseEvolutionaryAlgorithm:
    def __init__(self, population_size, num_meta_levels, genome_length, tournament_size, mutation_probability,
                 hierarchical_memory, function_decoder, input_data):
        self.population_size = population_size
        self.num_meta_levels = num_meta_levels
        self.genome_length = genome_length
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.hierarchical_memory = hierarchical_memory
        self.function_decoder = function_decoder
        self.input_data = input_data
        self.hierarchical_genome = HierarchicalGenome(num_meta_levels, genome_length, hierarchical_memory, function_decoder)
        self.fitness_cache = {}  # Cache to store fitness scores and evaluation flags

    def evolve(self, fitness_evaluator, num_generations, meta_epochs):
        for meta_epoch in range(meta_epochs):
            for level in range(self.num_meta_levels):
                print(f"Evolving Level {level}")
                population = self.hierarchical_genome.genomes[level]
                self.evaluate_level(level, population, fitness_evaluator)

                for generation in range(num_generations[level]):
                    # Tournament selection
                    parent = self.tournament_selection(population)
                    
                    # Create and mutate offspring
                    offspring = self.mutate(parent, level)
                    
                    # Add offspring to population and remove oldest member
                    population.append(offspring)
                    population.pop(0)
                    
                    # Mark offspring for re-evaluation
                    self.fitness_cache[self.genome_to_key(offspring)] = (None, True)
                    
                    # Update the level's population in the hierarchical genome
                    self.hierarchical_genome.genomes[level] = population
                    
                    # Calculate and print best fitness
                    best_fitness = max(self.get_fitness(genome, fitness_evaluator) for genome in population)
                    print(f"Meta-Epoch {meta_epoch +1} Generation {generation + 1}, Level {level}: Best Fitness = {best_fitness:.4f}")

                # Update lower_level_population for the next level with the whole population
                if level < self.num_meta_levels - 1:
                    for genome in self.hierarchical_genome.genomes[level + 1]:
                        genome.lower_level_population = population
                        # Invalidate cache for higher level genomes
                        self.fitness_cache[self.genome_to_key(genome)] = (None, True)

        return self.select_best_genomes(self.hierarchical_genome.genomes[-1], fitness_evaluator, num_best=1)[0]

    def mutate(self, genome, level):
        new_genome = FunctionGenome(
            self.genome_length,
            self.hierarchical_memory,
            self.function_decoder,
            meta_level=level,
            lower_level_population=self.hierarchical_genome.genomes[level-1] if level > 0 else None
        )
        new_genome.gene = genome.gene.copy()
        new_genome.input_gene = genome.input_gene.copy()
        new_genome.input_gene_2 = genome.input_gene_2.copy()
        new_genome.output_gene = genome.output_gene.copy()
        new_genome.mutate()
        return new_genome

    def evaluate_level(self, level, population, fitness_evaluator):
        for genome in population:
            self.get_fitness(genome, fitness_evaluator)

    def get_fitness(self, genome, fitness_evaluator):
        genome_key = self.genome_to_key(genome)
        if genome_key not in self.fitness_cache or self.fitness_cache[genome_key][1]:
            fitness = fitness_evaluator.evaluate(genome, self.input_data)
            if isinstance(fitness, (np.ndarray, list)):
                fitness = np.mean(fitness)  # Convert array to scalar
            self.fitness_cache[genome_key] = (fitness, False)
        return self.fitness_cache[genome_key][0]
    
    def genome_to_key(self, genome):
        # Include version numbers of lower-level genomes in the key
        lower_level_versions = tuple(g.version for g in genome.lower_level_population) if genome.lower_level_population else ()
        return (genome.version, tuple(genome.gene), tuple(genome.input_gene), 
                tuple(genome.input_gene_2), tuple(genome.output_gene), lower_level_versions)

    def tournament_selection(self, population):
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda genome: self.get_fitness(genome, self.fitness_evaluator))
    
    def select_best_genomes(self, population, fitness_evaluator, num_best=10):
        return sorted(population, key=lambda genome: self.get_fitness(genome, fitness_evaluator), reverse=True)[:num_best]

    
# Example usage in main script:
if __name__ == "__main__":
    # Initialize your components
    population_size = 100
    num_meta_levels = 2
    genome_length = 10
    tournament_size = 50
    mutation_probability = 1.0
    num_generations = [100,50,0]
    meta_epochs = 100

    num_scalars = 5
    num_vectors = 5
    num_tensors = 5
    scalar_size = 1
    vector_size = (3,)
    tensor_size = (3,3)
    
    hierarchical_memory = HierarchicalMemoryArrays(num_meta_levels, num_scalars, num_vectors, num_tensors, 
                                                   scalar_size, vector_size, tensor_size)
    function_decoder = FunctionDecoder()
    input_data = np.random.rand(3, 3)
    desired_outputs = input_data

    fitness_evaluator = FitnessEvaluator(desired_outputs)

    ea = LevelWiseEvolutionaryAlgorithm(
        population_size, num_meta_levels, genome_length, tournament_size,
        mutation_probability, hierarchical_memory, function_decoder, input_data
    )

    ea.fitness_evaluator = fitness_evaluator  # Add this line

    best_genome = ea.evolve(fitness_evaluator, num_generations,meta_epochs)

    print("\nBest Hierarchical Genome:")
    for level, genome in enumerate(best_genome.genomes):
        print(f"Level {level}: {genome.gene}")

    final_output = best_genome.execute_hierarchical(input_data)
    final_fitness = fitness_evaluator.calculate_fitness(final_output)
    print(f"\nFinal Fitness: {final_fitness:.4f}")
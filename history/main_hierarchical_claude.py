import torch
import random
import string
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np



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
                data = data.flatten()[-1]  # Take the last element if it's a tensor
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
        return np.divide(args[0], args[1])

    @staticmethod
    def abs_scalar(*args):
        return np.abs(args[0])

    @staticmethod
    def reciprocal_scalar(*args):
        return np.reciprocal(args[0])

    @staticmethod
    def sin_scalar(*args):
        return np.sin(args[0])

    @staticmethod
    def cos_scalar(*args):
        return np.cos(args[0])

    @staticmethod
    def tan_scalar(*args):
        return np.tan(args[0])

    @staticmethod
    def arcsin_scalar(*args):
        return np.arcsin(args[0])

    @staticmethod
    def arccos_scalar(*args):
        return np.arccos(args[0])

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
        return np.mod(args[0], args[1])

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
    def __init__(self, length, memory_arrays, function_decoder, meta_level=0):
        self.length = length
        self.memory_arrays = memory_arrays
        self.function_decoder = function_decoder
        self.meta_level = meta_level
        
        self.max_op = max(function_decoder.decoding_map.keys())
        self.gene = [random.randint(0, self.max_op) for _ in range(length)]
        self.memory_size = len(memory_arrays)
        self.input_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.input_gene_2 = [random.randint(0, self.memory_size - 1) for _ in range(length)]
        self.output_gene = [random.randint(0, self.memory_size - 1) for _ in range(length)]

    def __str__(self):
        return str(self.gene)

    def get_section(self, start, end):
        return self.gene[start:end]

    def get_input_position(self, index):
        return int(self.input_gene[index]) % len(self.memory_arrays)

    def get_input_position_2(self, index):
        return int(self.input_gene_2[index]) % len(self.memory_arrays)

    def get_output_position(self, index):
        return int(self.output_gene[index]) % len(self.memory_arrays)

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
            #output_gene_list = list(self.output_gene)
            self.output_gene[instruction_idx] = random.randint(0, self.memory_size - 1)

    def mutate_add_or_remove_one_instruction(self):
        instruction_idx = random.randint(0, self.length - 1)

        self.gene[instruction_idx] = random.randint(0, self.max_op)

        for gene_name in ['input_gene', 'input_gene_2', 'output_gene']:
            gene_list = getattr(self, gene_name)
            gene_list[instruction_idx] = random.randint(0, self.memory_size - 1)
            setattr(self, gene_name, gene_list)

    def mutate_all(self):
        
        self.gene = [random.randint(0, self.max_op) for _ in range(self.length)]
        self.input_gene = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]
        self.input_gene_2 = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]
        self.output_gene = [random.randint(0, self.memory_size - 1) for _ in range(self.length)]

    def mutate(self):
        mutation_type = random.choice(['one_argument', 'add_or_remove_one', 'all'])
        
        if mutation_type == 'one_argument':
            self.mutate_one_argument()
        elif mutation_type == 'add_or_remove_one':
            self.mutate_add_or_remove_one_instruction()
        else:  # mutation_type == 'all'
            self.mutate_all()

class HierarchicalGenome:
    def __init__(self, num_meta_levels, length_per_level, memory_arrays, function_decoder):
        self.num_meta_levels = num_meta_levels
        self.memory_arrays = memory_arrays
        self.function_decoder = function_decoder
        
        self.genomes = [
            FunctionGenome(length_per_level, memory_arrays, function_decoder, meta_level=i)
            for i in range(num_meta_levels)
        ]

    def mutate(self):
        # Randomly choose a meta-level to mutate
        meta_level = random.randint(0, self.num_meta_levels - 1)
        self.genomes[meta_level].mutate()

    def mutate_ops(self):
        # Mutate the operations (meta-level genomes)
        for genome in self.genomes[1:]:  # Skip meta-level 0
            if random.random() < 0.1:  # 10% chance to mutate each meta-level
                genome.mutate_all()

def meta_level_resolution(hierarchical_genome):
    resolved_genome = hierarchical_genome.genomes[0]  # Start with meta-level 0
    max_basic_op = max(resolved_genome.function_decoder.decoding_map.keys())
    
    for level in range(1, hierarchical_genome.num_meta_levels):
        current_genome = hierarchical_genome.genomes[level]
        new_resolved_genome = FunctionGenome(
            len(resolved_genome.gene) * len(current_genome.gene),
            resolved_genome.memory_arrays,
            resolved_genome.function_decoder
        )
        
        new_gene = []
        new_input_gene = []
        new_input_gene_2 = []
        new_output_gene = []
        
        for i, op in enumerate(current_genome.gene):
            if op <= max_basic_op:
                # If it's a basic operation, add it directly
                new_gene.append(op)
                new_input_gene.append(current_genome.input_gene[i])
                new_input_gene_2.append(current_genome.input_gene_2[i])
                new_output_gene.append(current_genome.output_gene[i])
            else:
                # If it's a higher-level operation, expand it
                index = op - max_basic_op - 1
                if 0 <= index < len(resolved_genome.gene):
                    new_gene.append(resolved_genome.gene[index])
                    new_input_gene.append(resolved_genome.input_gene[index])
                    new_input_gene_2.append(resolved_genome.input_gene_2[index])
                    new_output_gene.append(resolved_genome.output_gene[index])
        
        new_resolved_genome.gene = new_gene
        new_resolved_genome.input_gene = new_input_gene
        new_resolved_genome.input_gene_2 = new_input_gene_2
        new_resolved_genome.output_gene = new_output_gene
        
        resolved_genome = new_resolved_genome
    
    return resolved_genome

class CompositeFunction(torch.nn.Module):
    def __init__(self, functions, genome, memory_arrays):
        super(CompositeFunction, self).__init__()
        self.functions = functions
        self.genome = genome
        self.memory_arrays = memory_arrays

    def forward(self, input_data):
        # Set the input data into the memory arrays
        self.memory_arrays[-1] = input_data

        for i, function in enumerate(self.functions):
            input_position = self.genome.get_input_position(i)
            input_position_2 = self.genome.get_input_position_2(i)
            output_position = self.genome.get_output_position(i)

            input_data = self.memory_arrays[input_position]
            input_data_2 = self.memory_arrays[input_position_2]

            # Convert torch tensors to numpy arrays if necessary
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.numpy()
            if isinstance(input_data_2, torch.Tensor):
                input_data_2 = input_data_2.numpy()

            # Apply the function, always passing both inputs
            output_data = function(input_data, input_data_2)

            # Convert the output back to a torch tensor if necessary
            if isinstance(output_data, np.ndarray):
                output_data = torch.from_numpy(output_data).float()

            self.memory_arrays[output_position] = output_data

        return self.memory_arrays[0]  # Return the first value in memory as the final output

    def display(self):
        for i, func in enumerate(self.functions):
            input_pos = self.genome.get_input_position(i)
            input_pos_2 = self.genome.get_input_position_2(i)
            output_pos = self.genome.get_output_position(i)
            func_name = self.genome.gene[i]
            print(f"Step {i + 1}: {func_name}(memory[{input_pos}], memory[{input_pos_2}]) -> memory[{output_pos}]")    


class FitnessEvaluator:
    def __init__(self, desired_outputs, memory_arrays):
        self.desired_outputs = desired_outputs
        self.memory_arrays = memory_arrays

    def evaluate(self, composite_function, input_data):
        self.memory_arrays[0] = 0.0  # Reset the last memory cell to 0
        output = composite_function(input_data)
        desired_output = torch.tensor(self.desired_outputs).float()
        
        # Ensure output and desired_output have the same shape
        if output.shape != desired_output.shape:
            if len(desired_output.shape) == 0:
                output = output.flatten()[0]
            elif output.numel() == 1:
                output = output.expand(desired_output.shape)
            else:
                desired_output = desired_output.expand(output.shape)
        
        loss = torch.nn.functional.mse_loss(output, desired_output)
        fitness = 1 / (1 + loss.item())
        return fitness


class EvolutionaryAlgorithm:
    def __init__(self, population_size, num_meta_levels, genome_length_per_level, tournament_size, mutation_probability, memory_arrays, function_decoder, input_data):
        self.population_size = population_size
        self.num_meta_levels = num_meta_levels
        self.genome_length_per_level = genome_length_per_level
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.memory_arrays = memory_arrays
        self.function_decoder = function_decoder
        self.input_data = input_data
        self.population = [
            HierarchicalGenome(num_meta_levels, genome_length_per_level, memory_arrays, function_decoder)
            for _ in range(population_size)
        ]

    def evolve(self, fitness_evaluator, num_generations):
        fitness_history = []
        for generation in tqdm(range(num_generations)):
            # Evaluate fitness of each genome
            fitness_scores = [
                fitness_evaluator.evaluate(
                    CompositeFunction(
                        self.function_decoder.decode(meta_level_resolution(genome))[0],
                        meta_level_resolution(genome),
                        self.memory_arrays
                    ),
                    self.input_data
                )
                for genome in self.population
            ]

            # Select parents through tournament selection
            parents = [self.tournament_selection(fitness_scores) for _ in range(self.population_size)]

            # Create offspring through mutation
            offspring = [self.mutate(parents[i]) for i in range(self.population_size)]

            # Replace the old population with the offspring
            self.population = offspring

            # Find the best fitness score of the current generation
            best_index = fitness_scores.index(max(fitness_scores))
            best_fitness = fitness_scores[best_index]

            fitness_history.append(best_fitness)
            if generation % 10 == 0:
                print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")
                best_genome = self.population[best_index]
                print("Best Genome Structure:")
                for level, genome in enumerate(best_genome.genomes):
                    print(f"  Meta-level {level}: {genome.gene}")

        best_genome = self.population[fitness_scores.index(max(fitness_scores))]
        return best_genome, fitness_history

    def tournament_selection(self, fitness_scores):
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return self.population[winner_index]

    def mutate(self, genome):
        new_genome = HierarchicalGenome(
            self.num_meta_levels,
            self.genome_length_per_level,
            self.memory_arrays,
            self.function_decoder
        )
        
        # Copy the parent genome
        for i in range(self.num_meta_levels):
            new_genome.genomes[i].gene = genome.genomes[i].gene
            new_genome.genomes[i].input_gene = genome.genomes[i].input_gene
            new_genome.genomes[i].input_gene_2 = genome.genomes[i].input_gene_2
            new_genome.genomes[i].output_gene = genome.genomes[i].output_gene

        # Apply mutations
        if random.random() < self.mutation_probability:
            new_genome.mutate()  # Regular mutation
        
        if random.random() < self.mutation_probability:
            new_genome.mutate_ops()  # Mutation of meta-level operations

        return new_genome

# Example usage in main script:
if __name__ == '__main__':
    # Initialize components
    scalar_size = 1
    vector_size = (3,)
    tensor_size = (3, 3)
    num_scalars = 0
    num_vectors = 0
    num_tensors = 5
    memory_arrays = MemoryArrays(num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size)

    population_size = 100
    num_meta_levels = 3
    genome_length_per_level = 5
    tournament_size = 30
    mutation_probability = 0.1
    num_generations = 1000
    function_decoder = FunctionDecoder()
    input_data = torch.tensor([[1.0, 2.0, 3.0],[3.0, 23.0, 3.0],[3.0, 2.0, 3.0]])
    desired_outputs = input_data // 2

    evolutionary_algorithm = EvolutionaryAlgorithm(
        population_size,
        num_meta_levels,
        genome_length_per_level,
        tournament_size,
        mutation_probability,
        memory_arrays,
        function_decoder,
        input_data
    )

    fitness_evaluator = FitnessEvaluator(desired_outputs, memory_arrays)

    best_genome, fitness_history = evolutionary_algorithm.evolve(fitness_evaluator, num_generations)

    # Display the best genome
    print("\nBest Hierarchical Genome:")
    for level, genome in enumerate(best_genome.genomes):
        print(f"Meta-level {level}: {genome.gene}")

    # Resolve and evaluate the best genome
    resolved_genome = meta_level_resolution(best_genome)
    best_composite_function = CompositeFunction(
        function_decoder.decode(resolved_genome)[0],
        resolved_genome,
        memory_arrays
    )
    best_fitness = fitness_evaluator.evaluate(best_composite_function, input_data)
    print(f"\nBest Fitness: {best_fitness:.4f}")

    # Display the resolved genome
    print("\nResolved Genome:")
    best_composite_function.display()
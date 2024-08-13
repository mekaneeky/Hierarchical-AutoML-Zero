import random
import torch

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
        self.fitness = None  

    def function(self):
        # Create a copy of the genome's attributes to use in the closure
        gene = self.gene.copy()
        input_gene = self.input_gene.copy()
        input_gene_2 = self.input_gene_2.copy()
        output_gene = self.output_gene.copy()
        memory_size = self.memory_size
        function_decoder = self.function_decoder
        meta_level = self.meta_level
        lower_level_population = self.lower_level_population
        lower_level_size = self.lower_level_size

        def evolved_function(input_data, hierarchical_memory=None):
            # If no hierarchical_memory is provided, create a temporary one
            if hierarchical_memory is None:
                memory = [0] * memory_size
            else:
                memory = hierarchical_memory[meta_level]

            memory[-1] = input_data

            for i, op in enumerate(gene):
                if op in function_decoder.decoding_map:
                    func = function_decoder.decoding_map[op]
                    input1 = torch.tensor(memory[input_gene[i]])
                    input2 = torch.tensor(memory[input_gene_2[i]])
                    output = func(input1, input2)
                elif lower_level_population and op - len(function_decoder.decoding_map) < lower_level_size:
                    lower_genome = lower_level_population[op - len(function_decoder.decoding_map)]
                    output = lower_genome.function()(input_data, hierarchical_memory)
                else:
                    output = input_data  # No operation if op is out of range

                memory[output_gene[i]] = output

            return memory[0]  # Assuming the final output is always stored in the first memory location

        return evolved_function

    def execute(self, input_data):
        memory = self.hierarchical_memory[self.meta_level]
        memory[-1] = input_data

        for i, op in enumerate(self.gene):
            if op in self.function_decoder.decoding_map:
                func = self.function_decoder.decoding_map[op]
                input1 = torch.tensor(memory[self.input_gene[i]])
                input2 = torch.tensor(memory[self.input_gene_2[i]])
                output = func(input1, input2)
            elif self.lower_level_population and op - len(self.function_decoder.decoding_map) < self.lower_level_size:
                lower_genome = self.lower_level_population[op - len(self.function_decoder.decoding_map)]
                output = lower_genome.execute(input_data)
            else:
                output = input_data  # No operation if op is out of range

            memory[self.output_gene[i]] = output

        return memory[0]  # Assuming the final output is always stored in the first memory location

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
            output = lower_level_function.execute(input1, input2)

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
    def __init__(self, num_meta_levels, genome_length, hierarchical_memory, function_decoder, population_size):
        self.num_meta_levels = num_meta_levels
        self.hierarchical_memory = hierarchical_memory
        self.function_decoder = function_decoder

        self.genomes = [
            [FunctionGenome(genome_length, hierarchical_memory, function_decoder, meta_level=i)
             for _ in range(population_size)] for i in range(num_meta_levels)
        ]

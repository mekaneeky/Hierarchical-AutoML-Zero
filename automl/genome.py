import random
import torch
from copy import deepcopy
from .memory import CentralMemory

class FunctionGenome:
    def __init__(self, length, central_memory, function_decoder, meta_level=0, lower_level_population=None,input_addresses=None, output_addresses=None):
        self.length = length
        #self.central_memory = central_memory
        #self.memory = central_memory[meta_level]
        self.central_memory = central_memory
        self.memory = central_memory
        self.memory.reset()
        self.function_decoder = function_decoder
        self.meta_level = meta_level
        self.lower_level_population = lower_level_population
        self.SIGN_FLIP_PROB = 0.1


        self.expected_vector_size = deepcopy(self.memory.vector_memory[0].shape)
        self.expected_tensor_shape = deepcopy(self.memory.tensor_memory[0].shape)

        # Calculate size of the lower level population
        self.lower_level_size = len(lower_level_population) if lower_level_population else 0

        #max_op = max(function_decoder.decoding_map.keys())
        if meta_level > 0:
            self.max_op_pop = self.lower_level_size - 1
        else:
            self.max_op_pop = max(function_decoder.decoding_map.keys())

        self.gene = [random.randint(0, self.max_op_pop) for _ in range(length)]

        self.memory_size = len(self.memory)
        self.input_gene = []
        self.input_gene_2 = []
        self.output_gene = []
        for codon in self.gene:
            func, output_type, input_1_type, input_2_type = self.function_decoder.decoding_map[codon]
            self.input_gene.append(self.get_random_memory_address(input_1_type))
            self.input_gene_2.append(self.get_random_memory_address(input_2_type))
            self.output_gene.append(self.get_random_memory_address(output_type))
            
        self.constants_gene = [random.uniform(-1, 1) for _ in range(length)]
        self.constants_gene_2 = [random.uniform(0, 1) for _ in range(length)]

        ## For setting to matrix #assume square matrices
        #self.row_fixed = None
        #self.column_fixed = None
        self.set_row_column()
        
        self.version = 0
        self.fitness = None  
        self.input_addresses = input_addresses
        self.output_addresses = output_addresses


    def __deepcopy__(self, memo):
        # Create a new instance
        new_copy = FunctionGenome(self.length, self.central_memory, self.function_decoder, 
                                  self.meta_level, self.lower_level_population, input_addresses=self.input_addresses, output_addresses=self.output_addresses)
        
        # Copy simple attributes
        new_copy.gene = self.gene.copy()
        new_copy.input_gene = self.input_gene.copy()
        new_copy.input_gene_2 = self.input_gene_2.copy()
        new_copy.output_gene = self.output_gene.copy()
        new_copy.constants_gene = self.constants_gene.copy()
        new_copy.constants_gene_2 = self.constants_gene_2.copy()
        new_copy.version = self.version
        new_copy.fitness = self.fitness
        new_copy.memory = CentralMemory(
            num_scalars=len(self.memory.scalar_memory),
            num_vectors=len(self.memory.vector_memory),
            num_tensors=len(self.memory.tensor_memory),
            scalar_size=self.memory.scalar_memory[0].shape,
            vector_size=self.memory.vector_memory[0].shape,
            tensor_size=self.memory.tensor_memory[0].shape
        )


        return new_copy

    def validate_memory(self):
        if not hasattr(self.memory, 'vector_memory'):
            print("No vector memory")
            return False
        
        if len(self.memory.vector_memory) == 0:
            print("No vector memory")
            return False
        
        if not isinstance(self.memory.vector_memory[0], torch.Tensor):
            print("Vector memory not torch tensor")
            return False
        
        
        for i, vector in enumerate(self.memory.vector_memory):
            if vector.shape != self.expected_vector_size:
                breakpoint()
                print(f"Error: Inconsistent vector size at index {i}")
                print(f"Expected size: {self.expected_vector_size}, Actual size: {vector.shape[0]}")
                return False

        if not hasattr(self.memory, 'tensor_memory'):
            print("No tensor memory")
            return False
        
        if len(self.memory.tensor_memory) == 0:
            print("No tensor memory")
            return False
        
        
        for i, tensor in enumerate(self.memory.tensor_memory):
            if tensor.shape != self.expected_tensor_shape:
                breakpoint()
                print(f"Error: Inconsistent tensor shape at index {i}")
                print(f"Expected shape: {self.expected_tensor_shape}, Actual shape: {tensor.shape}")
                return False
        return True


    def set_row_column(self):

        self.row_fixed = random.randint(0,self.memory.vector_memory[0].shape[0]-1)
        self.column_fixed = random.randint(0, self.memory.tensor_memory[0].shape[1]-1)


    def get_random_memory_address(self, address_type):
        #address_type = random.choice(['scalar', 'vector', 'tensor'])
        if address_type == 'scalar':
            return self.memory.get_scalar_address()
        elif address_type == 'vector':
            return self.memory.get_vector_address()
        else:
            return self.memory.get_tensor_address()

    def function(self):
        # Create a copy of the genome's attributes to use in the closure
        gene = self.gene.copy()
        input_gene = self.input_gene.copy()
        input_gene_2 = self.input_gene_2.copy()
        output_gene = self.output_gene.copy()
        constants_gene = self.constants_gene.copy()
        constants_gene_2 = self.constants_gene_2.copy()
        memory_size = self.memory_size
        function_decoder = self.function_decoder
        meta_level = self.meta_level
        lower_level_population = self.lower_level_population
        lower_level_size = self.lower_level_size
        self.memory.reset()
        memory = self.memory

        def evolved_function(*args):
            # If no central_memory is provided, create a temporary one
            #if central_memory is None:
            #    memory = [0] * memory_size
            #else:
            #    memory = central_memory[meta_level]

            for i, data in enumerate(args):
                self.memory[self.input_addresses[i]] = data

            for i, op in enumerate(gene):
                if meta_level == 0:
                    func = function_decoder.decoding_map[op][0]
                    input1 = torch.tensor(memory[input_gene[i]])
                    input2 = torch.tensor(memory[input_gene_2[i]])
                    constant = torch.tensor(constants_gene[i])
                    constant_2 = torch.tensor(constants_gene_2[i])
                    output = func(input1, input2,constant, constant_2, self.row_fixed, self.column_fixed)

                else:
                    lower_genome = lower_level_population[op]
                    output = lower_genome.function()(*args)
                
                #if not self.validate_memory():
                #    print("Premystery")
                #    breakpoint()
                memory[output_gene[i]] = output
                #if not self.validate_memory():
                #    print("Postmystery")
                #    breakpoint()
            if len(self.output_addresses) == 1:
                return memory[self.output_addresses[0]]
            
            return tuple(self.memory[addr] for addr in self.output_addresses)  # Assuming the final output is always stored in the first memory location

        return evolved_function

    def execute(self, *args):
        memory = self.central_memory[self.meta_level]
        
        for i, data in enumerate(args):
            self.memory[self.input_addresses[i]] = data

        for i, op in enumerate(self.gene):
            if self.meta_level == 0:
                func = self.function_decoder.decoding_map[op]
                input1 = torch.tensor(memory[self.input_gene[i]])
                input2 = torch.tensor(memory[self.input_gene_2[i]])
                constant = torch.tensor(self.constants_gene[i])
                constant_2 = torch.tensor(self.constants_gene_2[i])
                output = func(input1, input2, constant, constant_2, self.row_fixed, self.column_fixed )
            else:
                lower_genome = self.lower_level_population[op]
                output = lower_genome.execute(*args)
            

            memory[self.output_gene[i]] = output
        
        if len(self.output_addresses) == 1:
            return memory[self.output_addresses[0]]
        
        return tuple(self.memory[addr] for addr in self.output_addresses)  # Assuming the final output is always stored in the first memory location

    def mutate(self):
        mutation_type = random.choice(['one_argument', 'add_or_remove_one', 'all', 'constant'])

        if mutation_type == 'one_argument':
            self.mutate_one_argument()
        elif mutation_type == 'add_or_remove_one':
            self.mutate_add_or_remove_one_instruction()
        elif mutation_type == 'constant':
            self.mutate_constant()
        else:  # mutation_type == 'all'
            self.mutate_all()
        self.version += 1

    @staticmethod
    def mutate_float_log_scale(value): 
        """Mutate a float value using log-scale mutation.""" 
        if value > 0: 
            return torch.exp(torch.log(torch.tensor(value)) + random.gauss(0.0, 1.0)) 
        else: 
            return -torch.exp(torch.log(-torch.tensor(value)) + random.gauss(0.0, 1.0)) 

    def mutate_constant(self): 
        """Either flip the sign of a value or apply log-scale mutation.""" 
        instruction_idx = random.randint(0, self.length - 1)
        
        if random.random() < 0.5:
            gene_to_mutate = self.constants_gene
        else:
            gene_to_mutate = self.constants_gene_2
            if random.random() < self.SIGN_FLIP_PROB: 
                gene_to_mutate[instruction_idx] = -gene_to_mutate[instruction_idx] 
            else: 
                gene_to_mutate[instruction_idx] = self.mutate_float_log_scale(gene_to_mutate[instruction_idx])

    def mutate_one_argument(self):
        instruction_idx = random.randint(0, self.length - 1)
        func, output_type, input_1_type, input_2_type = self.function_decoder.decoding_map[self.gene[instruction_idx]]
        if random.random() > 0.5:
            # Mutate input argument
            gene_to_mutate = random.choice(['input_gene', 'input_gene_2'])
            if gene_to_mutate == "input_gene":
                self.input_gene[instruction_idx] = self.get_random_memory_address(input_1_type)
            else: #input_gene_2
                self.input_gene_2[instruction_idx] = self.get_random_memory_address(input_2_type)
        else:
            # Mutate output argument
            self.output_gene[instruction_idx] = self.get_random_memory_address(output_type)

    def mutate_add_or_remove_one_instruction(self):
        instruction_idx = random.randint(0, self.length - 1)

        self.gene[instruction_idx] = random.randint(0, self.max_op_pop)
        func, output_type, input_1_type, input_2_type = self.function_decoder.decoding_map[self.gene[instruction_idx]]

        self.input_gene[instruction_idx] = self.get_random_memory_address(input_1_type)
        self.input_gene_2[instruction_idx] = self.get_random_memory_address(input_2_type)
        self.output_gene[instruction_idx] = self.get_random_memory_address(output_type)

    def mutate_all(self):
        self.gene = [random.randint(0, self.max_op_pop) for _ in range(self.length)]

        for codon in self.gene:
            func, output_type, input_1_type, input_2_type = self.function_decoder.decoding_map[codon]
            self.input_gene.append(self.get_random_memory_address(input_1_type))
            self.input_gene_2.append(self.get_random_memory_address(input_2_type))
            self.output_gene.append(self.get_random_memory_address(output_type))

        self.constants_gene = [random.uniform(-1, 1) for _ in range(self.length)]
        self.constants_gene_2 = [random.uniform(0, 1) for _ in range(self.length)]

class TopKFunctionGenome(FunctionGenome):
    def __init__(self):
        raise NotImplementedError

class FreeForAllFunctionGenome(FunctionGenome):
    def __init__(self, length, central_memory, function_decoder, meta_level=0, lower_level_population=None):
        self.length = length
        self.central_memory = central_memory
        self.function_decoder = function_decoder
        self.meta_level = meta_level
        self.lower_level_population = lower_level_population

        # Calculate size of the lower level population
        self.lower_level_size = len(lower_level_population) if lower_level_population else 0

        max_op = max(function_decoder.decoding_map.keys())
        self.max_op_pop = max_op + self.lower_level_size #TODO add alternative that works per genome only
        #TODO make all level function support an optional case
        self.gene = [random.randint(0, self.max_op_pop) for _ in range(length)]

        self.memory_size = len(central_memory[meta_level])
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

        def evolved_function(input_data, central_memory=None):
            # If no central_memory is provided, create a temporary one
            if central_memory is None:
                memory = [0] * memory_size
            else:
                memory = central_memory[meta_level]

            memory[-1] = input_data

            for i, op in enumerate(gene):
                if op in function_decoder.decoding_map:
                    func = function_decoder.decoding_map[op]
                    input1 = torch.tensor(memory[input_gene[i]])
                    input2 = torch.tensor(memory[input_gene_2[i]])
                    output = func(input1, input2)
                elif lower_level_population and op - len(function_decoder.decoding_map) < lower_level_size:
                    lower_genome = lower_level_population[op - len(function_decoder.decoding_map)]
                    output = lower_genome.function()(input_data, central_memory)
                else:
                    output = input_data  # No operation if op is out of range

                memory[output_gene[i]] = output

            return memory[0]  # Assuming the final output is always stored in the first memory location

        return evolved_function

    def execute(self, input_data):
        memory = self.central_memory[self.meta_level]
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
                raise ValueError("OP is out of range")
                output = input_data  # No operation if op is out of range

            memory[self.output_gene[i]] = output

        return memory[0]  # Assuming the final output is always stored in the first memory location



class HierarchicalFunctionGenome:
    def __init__(self, length, central_memory, meta_level, lower_level_population):
        self.length = length
        self.central_memory = central_memory
        self.meta_level = meta_level
        self.lower_level_population = lower_level_population

        self.memory_size = len(central_memory[meta_level])

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
        memory = self.central_memory[self.meta_level]
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
        new_copy = HierarchicalFunctionGenome(self.length, self.central_memory, self.meta_level, self.lower_level_population)

        # Copy the genes
        new_copy.gene = self.gene.copy()
        new_copy.input_gene = self.input_gene.copy()
        new_copy.input_gene_2 = self.input_gene_2.copy()
        new_copy.output_gene = self.output_gene.copy()

        return new_copy

class HierarchicalGenome:
    def __init__(self, num_meta_levels, genome_length, central_memory, function_decoder, population_size, input_addresses, output_addresses):
        self.num_meta_levels = num_meta_levels
        self.central_memory = central_memory
        self.function_decoder = function_decoder
        self.input_addresses = input_addresses
        self.output_addresses = output_addresses
        self.genomes = []
        current_population = []
        for meta_level in range(num_meta_levels):
            
            for _ in range(population_size):
                if meta_level == 0: 
                    current_population.append(FunctionGenome(genome_length, central_memory, function_decoder, meta_level=meta_level, lower_level_population=None, input_addresses=self.input_addresses, output_addresses=self.output_addresses))
                else:
                    current_population.append(FunctionGenome(genome_length, central_memory, function_decoder, meta_level=meta_level, lower_level_population=self.genomes[-1],input_addresses=self.input_addresses, output_addresses=self.output_addresses))
            self.genomes.append(deepcopy(current_population))
            current_population = []

        


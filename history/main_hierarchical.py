import numpy as np
import random
import string

class OPGenome:
    def __init__(self, metalevel_count, number_of_meta_ops, prior_level_ops):
        self.metalevel_count = metalevel_count
        self.number_of_meta_ops = number_of_meta_ops
        self.prior_level_ops = prior_level_ops
        self.combine_OPs = self.create_OP_gene()

    def create_OP_gene(self):
        return np.random.randint(0, self.number_of_meta_ops, 
                                 size=(self.metalevel_count, self.number_of_meta_ops, self.prior_level_ops))

    def mutate(self, mutation_rate=0.01):
        mask = np.random.random(self.combine_OPs.shape) < mutation_rate
        self.combine_OPs[mask] = np.random.randint(0, self.number_of_meta_ops, size=np.sum(mask))

class HierarchicalGenome:
    def __init__(self, length, memory_arrays, function_decoder, num_meta_levels):
        self.length = length
        self.memory_arrays = memory_arrays
        self.function_decoder = function_decoder
        self.num_meta_levels = num_meta_levels
        
        self.op_genome = OPGenome(num_meta_levels, len(function_decoder.decoding_map), 2)  # Assuming binary operations
        self.gene = self.generate_random_hierarchical_gene()

    def generate_random_hierarchical_gene(self):
        valid_keys = ''.join(self.function_decoder.decoding_map.keys())
        
        base_OP = ''.join(random.choices(valid_keys, k=self.length))
        OP_metalevel = ''.join(random.choices(string.digits[:self.num_meta_levels], k=self.length))
        input_gene = ''.join(random.choices(string.digits, k=self.length))
        input_gene_2 = ''.join(random.choices(string.digits, k=self.length))
        output_gene = ''.join(random.choices(string.digits, k=self.length))
        
        return base_OP, OP_metalevel, input_gene, input_gene_2, output_gene

    def mutate(self):
        mutation_type = random.choice(['base_OP', 'OP_metalevel', 'input', 'input_2', 'output', 'op_genome'])
        
        if mutation_type == 'base_OP':
            index = random.randint(0, self.length - 1)
            self.gene[0] = self.gene[0][:index] + random.choice(list(self.function_decoder.decoding_map.keys())) + self.gene[0][index+1:]
        elif mutation_type == 'OP_metalevel':
            index = random.randint(0, self.length - 1)
            self.gene[1] = self.gene[1][:index] + random.choice(string.digits[:self.num_meta_levels]) + self.gene[1][index+1:]
        elif mutation_type in ['input', 'input_2', 'output']:
            gene_index = 2 if mutation_type == 'input' else (3 if mutation_type == 'input_2' else 4)
            index = random.randint(0, self.length - 1)
            self.gene[gene_index] = self.gene[gene_index][:index] + random.choice(string.digits) + self.gene[gene_index][index+1:]
        else:  # op_genome
            self.op_genome.mutate()

    def get_input_position(self, index):
        return int(self.gene[2][index]) % len(self.memory_arrays)

    def get_input_position_2(self, index):
        return int(self.gene[3][index]) % len(self.memory_arrays)

    def get_output_position(self, index):
        return int(self.gene[4][index]) % len(self.memory_arrays)

    def get_base_op(self, index):
        return self.gene[0][index]

    def get_meta_level(self, index):
        return int(self.gene[1][index])

    def display(self):
        print("Hierarchical Genome Structure:")
        for i in range(self.length):
            print(f"Step {i+1}: Base OP: {self.get_base_op(i)}, Meta-level: {self.get_meta_level(i)}, "
                  f"Input 1: {self.get_input_position(i)}, Input 2: {self.get_input_position_2(i)}, "
                  f"Output: {self.get_output_position(i)}")
        
        print("\nOP Genome Structure:")
        for level in range(self.num_meta_levels):
            print(f"Meta-level {level}:")
            for op in range(len(self.function_decoder.decoding_map)):
                print(f"  OP {chr(op + ord('A'))}: {self.op_genome.combine_OPs[level, op]}")

# Example usage in CompositeFunction
class CompositeFunction(torch.nn.Module):
    def __init__(self, hierarchical_genome, memory_arrays):
        super(CompositeFunction, self).__init__()
        self.genome = hierarchical_genome
        self.memory_arrays = memory_arrays

    def forward(self, input_data):
        # Set the input data into the memory arrays
        self.memory_arrays[-1] = input_data

        for i in range(self.genome.length):
            base_op = self.genome.get_base_op(i)
            meta_level = self.genome.get_meta_level(i)
            input_pos = self.genome.get_input_position(i)
            input_pos_2 = self.genome.get_input_position_2(i)
            output_pos = self.genome.get_output_position(i)

            input_data = self.memory_arrays[input_pos]
            input_data_2 = self.memory_arrays[input_pos_2]

            # Convert torch tensors to numpy arrays if necessary
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.numpy()
            if isinstance(input_data_2, torch.Tensor):
                input_data_2 = input_data_2.numpy()

            # Apply the function, always passing both inputs
            function = self.genome.function_decoder.decoding_map[base_op]
            output_data = function(input_data, input_data_2)

            # Convert the output back to a torch tensor if necessary
            if isinstance(output_data, np.ndarray):
                output_data = torch.from_numpy(output_data).float()

            self.memory_arrays[output_pos] = output_data

        return self.memory_arrays[0]  # Return the first value in memory as the final output

    def display(self):
        self.genome.display()
class HierarchicalGenomeUnpacker:
    def __init__(self):
        self.op_templates = {
            0: "memory[{out}] = memory[{in1}]  # DO NOTHING",
            1: "memory[{out}] = torch.add(memory[{in1}], memory[{in2}])  # ADD",
            2: "memory[{out}] = torch.subtract(memory[{in1}], memory[{in2}])  # SUBTRACT",
            3: "memory[{out}] = torch.multiply(memory[{in1}], memory[{in2}])  # MULTIPLY",
            4: "memory[{out}] = torch.divide(memory[{in1}], memory[{in2}])  # DIVIDE",
            5: "memory[{out}] = torch.abs(memory[{in1}])  # ABS",
            6: "memory[{out}] = torch.reciprocal(memory[{in1}])  # RECIPROCAL",
            7: "memory[{out}] = torch.sin(memory[{in1}])  # SIN",
            8: "memory[{out}] = torch.cos(memory[{in1}])  # COS",
            9: "memory[{out}] = torch.tan(memory[{in1}])  # TAN",
            10: "memory[{out}] = torch.arcsin(memory[{in1}])  # ARCSIN",
            11: "memory[{out}] = torch.arccos(memory[{in1}])  # ARCCOS",
            12: "memory[{out}] = torch.arctan(memory[{in1}])  # ARCTAN",
            13: "memory[{out}] = 1 / (1 + torch.exp(-memory[{in1}]))  # SIGMOID",
            14: "memory[{out}] = torch.where(memory[{in1}] > 0, memory[{in1}], memory[{in1}] * 0.01)  # LEAKY RELU",
            15: "memory[{out}] = torch.relu(memory[{in1}])  # RELU",
            16: "memory[{out}] = self.stable_softmax(memory[{in1}])  # STABLE SOFTMAX",
            17: "memory[{out}] = torch.mean(memory[{in1}], axis=0)  # MEAN AXIS",
            18: "memory[{out}] = torch.std(memory[{in1}], axis=0)  # STD AXIS",
            19: "memory[{out}] = {in1}  # SET CONSTANT SCALAR",
            20: "memory[{out}] = torch.log(torch.abs(memory[{in1}]) + 1e-10)  # LOG",
            21: "memory[{out}] = torch.pow(memory[{in1}], memory[{in2}])  # POWER",
            22: "memory[{out}] = torch.sqrt(torch.abs(memory[{in1}]))  # SQRT",
            23: "memory[{out}] = torch.maximum(memory[{in1}], memory[{in2}])  # MAX",
            24: "memory[{out}] = torch.minimum(memory[{in1}], memory[{in2}])  # MIN",
            25: "memory[{out}] = torch.mod(memory[{in1}], memory[{in2}])  # MOD",
            26: "memory[{out}] = torch.sign(memory[{in1}])  # SIGN",
            27: "memory[{out}] = torch.floor(memory[{in1}])  # FLOOR",
            28: "memory[{out}] = torch.ceil(memory[{in1}])  # CEIL",
            29: "memory[{out}] = torch.round(memory[{in1}])  # ROUND",
            30: "memory[{out}] = torch.hypot(memory[{in1}], memory[{in2}])  # HYPOT",
        }
        self.cache = {}  # New cache attribute

    def unpack_function_genome(self, genome, function_name="execute_function_genome"):
        code_lines = ["import torch", ""]
        
        if genome.meta_level == 0:
            code_lines.extend(self._unpack_base_genome(genome, function_name))
        else:
            code_lines.extend(self._unpack_higher_level_genome(genome.meta_level, 0, genome, function_name))
            
            # Unpack lower-level genomes
            for i, lower_genome in enumerate(genome.lower_level_population):
                lower_function_name = f"execute_level_{genome.meta_level-1}_genome_{i}"
                code_lines.extend(self.unpack_function_genome(lower_genome, lower_function_name))
        
        return "\n".join(code_lines)

    def unpack_hierarchical_genome(self, hierarchical_genome, gene_idx, selective=False):
        self.cache = {}  # Reset cache for each unpacking operation
        code_lines = ["import torch", ""]
        
        if selective:
            top_level = hierarchical_genome.num_meta_levels - 1
            gene_to_expand = hierarchical_genome.genomes[top_level][gene_idx]
            code_lines.extend(self.unpack_function_genome(gene_to_expand, f"execute_level_{top_level}_genome_{gene_idx}"))
            
            for sub_gene_idx in gene_to_expand.gene:
                lower_genome = hierarchical_genome.genomes[top_level-1][sub_gene_idx]
                code_lines.extend(self.unpack_function_genome(lower_genome, f"execute_level_{top_level-1}_genome_{sub_gene_idx}"))
        else:
            for level in range(hierarchical_genome.num_meta_levels):
                for i, genome in enumerate(hierarchical_genome.genomes[level]):
                    code_lines.extend(self.unpack_function_genome(genome, f"execute_level_{level}_genome_{i}"))
        
        code_lines.extend(self._create_main_function(hierarchical_genome, gene_idx, selective))
        
        return "\n".join(code_lines)
    
    
    def _unpack_selective(self, hierarchical_genome, level, genome_index):
        code_lines = []
        cache_key = (level, genome_index)
        if cache_key in self.cache:
            return code_lines  # Return empty list if already cached to avoid duplication
        
        if level < 0 or level >= len(hierarchical_genome.genomes):
            print(f"Warning: Invalid level {level}. Skipping.")
            return code_lines

        if genome_index < 0 or genome_index >= len(hierarchical_genome.genomes[level]):
            print(f"Warning: Invalid genome index {genome_index} at level {level}. Skipping.")
            return code_lines

        genome = hierarchical_genome.genomes[level][genome_index]
        function_name = f"execute_level_{level}_genome_{genome_index}"
        
        if level == 0:
            code_lines.extend(self._unpack_base_genome(genome, function_name))
        else:
            code_lines.extend(self._unpack_higher_level_genome(level, genome_index, genome, function_name))
            
            for lower_genome_index in genome.gene:
                if 0 <= lower_genome_index < len(hierarchical_genome.genomes[level-1]):
                    code_lines.extend(self._unpack_selective(hierarchical_genome, level - 1, lower_genome_index))
                else:
                    print(f"Warning: Invalid lower genome index {lower_genome_index} at level {level-1}. Skipping.")
        
        self.cache[cache_key] = True  # Mark as cached
        return code_lines


    def _unpack_level(self, level, genomes):
        code_lines = [f"# Level {level} functions"]
        
        for i, genome in enumerate(genomes):
            function_name = f"execute_level_{level}_genome_{i}"
            code_lines.extend(self._unpack_genome(level, i, genome, function_name))
            code_lines.append("")
        
        return code_lines

    def _unpack_genome(self, level, genome_index, genome, function_name):
        if level == 0:
            return self._unpack_base_genome(genome, function_name)
        else:
            return self._unpack_higher_level_genome(level, genome_index, genome, function_name)

    def _unpack_base_genome(self, genome, function_name):
        code_lines = [f"def {function_name}(memory):"]
        
        for i, op in enumerate(genome.gene):
            if op in self.op_templates:
                template = self.op_templates[op]
                line = "    " + template.format(out=genome.output_gene[i], in1=genome.input_gene[i], in2=genome.input_gene_2[i])
                code_lines.append(line)
            else:
                code_lines.append(f"    # Unknown operation: {op}")
        
        code_lines.append("    return memory[0]  # Assuming output is in first memory location\n")
        
        return code_lines

    def _unpack_higher_level_genome(self, level, genome_index, genome, function_name):
        code_lines = [f"def {function_name}(memory_level_{level}, memory_level_{level-1}):"]
        
        for i, function_index in enumerate(genome.gene):
            if 0 <= function_index < len(genome.lower_level_population):
                lower_level_function = f"execute_level_{level-1}_genome_{function_index}"
                line = f"    memory_level_{level}[{genome.output_gene[i]}] = {lower_level_function}(memory_level_{level-1})"
                code_lines.append(line)
            else:
                print(f"Warning: Invalid function index {function_index} in genome {genome_index} at level {level}. Skipping.")
        
        code_lines.append(f"    return memory_level_{level}[0]  # Assuming output is in first memory location\n")
        
        return code_lines

    def _create_main_function(self, hierarchical_genome, gene_idx, selective):
        code_lines = ["def execute_hierarchical_genome(input_data, hierarchical_memory):"]
        code_lines.append("    result = input_data")
        
        top_level = hierarchical_genome.num_meta_levels - 1
        code_lines.append(f"    # Level {top_level}")
        
        if top_level == 0:
            code_lines.append(f"    memory = hierarchical_memory[{top_level}]")
            code_lines.append(f"    memory[-1] = result  # Store input data")
            code_lines.append(f"    result = execute_level_{top_level}_genome_{gene_idx}(memory)")
        else:
            code_lines.append(f"    memory_level_{top_level} = hierarchical_memory[{top_level}]")
            code_lines.append(f"    memory_level_{top_level-1} = hierarchical_memory[{top_level-1}]")
            top_genome = hierarchical_genome.genomes[top_level][gene_idx]
            for i, lower_genome_index in enumerate(top_genome.gene):
                code_lines.append(f"    result = execute_level_{top_level-1}_genome_{lower_genome_index}(memory_level_{top_level}, memory_level_{top_level-1})")
                code_lines.append(f"    memory_level_{top_level-1}[{top_genome.output_gene[i]}] = result")
        
        code_lines.append("    return result\n")
        
        return code_lines
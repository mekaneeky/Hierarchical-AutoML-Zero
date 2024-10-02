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
            16: "memory[{out}] = {constant}  # SET CONSTANT SCALAR",
            17: "memory[{out}] = torch.empty(1).normal_({in1}, {in2})  # GAUSSIAN SCALAR",
            18: "memory[{out}] = torch.empty(memory[{in1}].shape).normal_({in1}, {in2})  # GAUSSIAN MATRIX",
            19: "memory[{out}] = torch.empty(memory[{in1}].shape).uniform_(memory[{in2}], memory[CONSTANT])  # UNIFORM SCALAR",
            20: "memory[{out}] = torch.log(torch.abs(memory[{in1}]) + 1e-10)  # LOG",
            21: "memory[{out}] = torch.pow(memory[{in1}], memory[{in2}])  # POWER",
            22: "memory[{out}] = torch.sqrt(torch.abs(memory[{in1}]))  # SQRT",
            23: "memory[{out}] = torch.maximum(memory[{in1}], memory[{in2}])  # MAX SCALAR",
            24: "memory[{out}] = torch.minimum(memory[{in1}], memory[{in2}])  # MIN SCALAR",
            25: "memory[{out}] = torch.remainder(memory[{in1}], memory[{in2}])  # MOD",
            26: "memory[{out}] = torch.sign(memory[{in1}])  # SIGN",
            27: "memory[{out}] = torch.floor(memory[{in1}])  # FLOOR",
            28: "memory[{out}] = torch.ceil(memory[{in1}])  # CEIL",
            29: "memory[{out}] = torch.round(memory[{in1}])  # ROUND",
            30: "memory[{out}] = torch.hypot(memory[{in1}], memory[{in2}])  # HYPOT",
            31: "memory[{out}] = torch.add(memory[{in1}], memory[{in2}])  # ADD VECTOR",
            32: "memory[{out}] = torch.subtract(memory[{in1}], memory[{in2}])  # SUBTRACT VECTOR",
            33: "memory[{out}] = torch.multiply(memory[{in1}], memory[{in2}])  # MULTIPLY VECTOR",
            34: "memory[{out}] = torch.divide(memory[{in1}], memory[{in2}])  # DIVIDE VECTOR",
            35: "memory[{out}] = torch.abs(memory[{in1}])  # ABS VECTOR",
            36: "memory[{out}] = torch.reciprocal(memory[{in1}])  # RECIPROCAL VECTOR",
            37: "memory[{out}] = torch.sin(memory[{in1}])  # SIN VECTOR",
            38: "memory[{out}] = torch.cos(memory[{in1}])  # COS VECTOR",
            39: "memory[{out}] = torch.tan(memory[{in1}])  # TAN VECTOR",
            40: "memory[{out}] = torch.arcsin(memory[{in1}])  # ARCSIN VECTOR",
            41: "memory[{out}] = torch.arccos(memory[{in1}])  # ARCCOS VECTOR",
            42: "memory[{out}] = torch.arctan(memory[{in1}])  # ARCTAN VECTOR",
            43: "memory[{out}] = 1 / (1 + torch.exp(-memory[{in1}]))  # SIGMOID VECTOR",
            44: "memory[{out}] = torch.where(memory[{in1}] > 0, memory[{in1}], memory[{in1}] * 0.01)  # LEAKY RELU VECTOR",
            45: "memory[{out}] = torch.relu(memory[{in1}])  # RELU VECTOR",
            46: "memory[{out}] = self.stable_softmax(memory[{in1}])  # STABLE SOFTMAX VECTOR",
            47: "memory[{out}] = torch.mean(memory[{in1}])  # MEAN VECTOR",
            48: "memory[{out}] = torch.std(memory[{in1}])  # STD VECTOR",
            49: "memory[{out}] = torch.empty(memory[{in1}].shape).uniform_(memory[{in2}], memory[CONSTANT])  # UNIFORM VECTOR",
            50: "memory[{out}] = torch.log(torch.abs(memory[{in1}]) + 1e-10)  # LOG VECTOR",
            51: "memory[{out}] = torch.pow(memory[{in1}], memory[{in2}])  # POWER VECTOR",
            52: "memory[{out}] = torch.sqrt(torch.abs(memory[{in1}]))  # SQRT VECTOR",
            53: "memory[{out}] = torch.max(memory[{in1}])  # MAX VECTOR",
            54: "memory[{out}] = torch.min(memory[{in1}])  # MIN VECTOR",
            55: "memory[{out}] = torch.remainder(memory[{in1}], memory[{in2}])  # MOD VECTOR",
            56: "memory[{out}] = torch.sign(memory[{in1}])  # SIGN VECTOR",
            57: "memory[{out}] = torch.floor(memory[{in1}])  # FLOOR VECTOR",
            58: "memory[{out}] = torch.ceil(memory[{in1}])  # CEIL VECTOR",
            59: "memory[{out}] = torch.round(memory[{in1}])  # ROUND VECTOR",
            60: "memory[{out}] = torch.hypot(memory[{in1}], memory[{in2}])  # HYPOT VECTOR",
            61: "memory[{out}] = torch.dot(memory[{in1}], memory[{in2}])  # DOT VECTOR",
            62: "memory[{out}] = torch.norm(memory[{in1}])  # NORM VECTOR",
            63: "memory[{out}] = torch.add(memory[{in1}], memory[{in2}])  # ADD MATRIX",
            64: "memory[{out}] = torch.subtract(memory[{in1}], memory[{in2}])  # SUBTRACT MATRIX",
            65: "memory[{out}] = torch.multiply(memory[{in1}], memory[{in2}])  # MULTIPLY MATRIX",
            66: "memory[{out}] = torch.divide(memory[{in1}], memory[{in2}])  # DIVIDE MATRIX",
            67: "memory[{out}] = torch.abs(memory[{in1}])  # ABS MATRIX",
            68: "memory[{out}] = torch.reciprocal(memory[{in1}])  # RECIPROCAL MATRIX",
            69: "memory[{out}] = torch.sin(memory[{in1}])  # SIN MATRIX",
            70: "memory[{out}] = torch.cos(memory[{in1}])  # COS MATRIX",
            71: "memory[{out}] = torch.tan(memory[{in1}])  # TAN MATRIX",
            72: "memory[{out}] = torch.arcsin(memory[{in1}])  # ARCSIN MATRIX",
            73: "memory[{out}] = torch.arccos(memory[{in1}])  # ARCCOS MATRIX",
            74: "memory[{out}] = torch.arctan(memory[{in1}])  # ARCTAN MATRIX",
            75: "memory[{out}] = 1 / (1 + torch.exp(-memory[{in1}]))  # SIGMOID MATRIX",
            76: "memory[{out}] = torch.where(memory[{in1}] > 0, memory[{in1}], memory[{in1}] * 0.01)  # LEAKY RELU MATRIX",
            77: "memory[{out}] = torch.relu(memory[{in1}])  # RELU MATRIX",
            78: "memory[{out}] = self.stable_softmax(memory[{in1}])  # STABLE SOFTMAX MATRIX",
            79: "memory[{out}] = torch.mean(memory[{in1}])  # MEAN MATRIX",
            80: "memory[{out}] = torch.std(memory[{in1}], dim=0)  # STD MATRIX",
            81: "memory[{out}] = torch.empty(memory[{in1}].shape).uniform_(memory[{in2}], memory[CONSTANT])  # UNIFORM MATRIX",
            82: "memory[{out}] = torch.log(torch.abs(memory[{in1}]) + 1e-10)  # LOG MATRIX",
            83: "memory[{out}] = torch.pow(memory[{in1}], memory[{in2}])  # POWER MATRIX",
            84: "memory[{out}] = torch.sqrt(torch.abs(memory[{in1}]))  # SQRT MATRIX",
            85: "memory[{out}] = torch.max(memory[{in1}])  # MAX MATRIX",
            86: "memory[{out}] = torch.min(memory[{in1}])  # MIN MATRIX",
            87: "memory[{out}] = torch.remainder(memory[{in1}], memory[{in2}])  # MOD MATRIX",
            88: "memory[{out}] = torch.sign(memory[{in1}])  # SIGN MATRIX",
            89: "memory[{out}] = torch.floor(memory[{in1}])  # FLOOR MATRIX",
            90: "memory[{out}] = torch.ceil(memory[{in1}])  # CEIL MATRIX",
            91: "memory[{out}] = torch.round(memory[{in1}])  # ROUND MATRIX",
            92: "memory[{out}] = torch.hypot(memory[{in1}], memory[{in2}])  # HYPOT MATRIX",
            93: "memory[{out}] = torch.norm(memory[{in1}])  # NORM MATRIX",
            94: "memory[{out}] = torch.multiply(memory[{in1}], memory[{in2}])  # MULTIPLY VECTOR BY SCALAR",
            95: "memory[{out}] = torch.multiply(memory[{in1}], memory[{in2}])  # MULTIPLY MATRIX BY SCALAR",
            96: "memory[{out}] = torch.full((memory[{in2}].shape[0],), memory[{in1}].item())  # BROADCAST SCALAR TO VECTOR",
            97: "memory[{out}] = memory[{in1}].repeat(memory[{in2}].shape[0], 1)  # BROADCAST VECTOR TO MATRIX ROW",
            98: "memory[{out}] = memory[{in1}].repeat(memory[{in2}].shape[1], 1).t()  # BROADCAST VECTOR TO MATRIX COL",
            99: "memory[{out}] = torch.outer(memory[{in1}], memory[{in2}])  # OUTER PRODUCT",
            100: "memory[{out}] = torch.matmul(memory[{in1}], memory[{in2}])  # MATMUL",
            101: "memory[{out}] = memory[{in1}].t()  # TRANSPOSE",
            102: "memory[{out}] = torch.empty(memory[{in1}].shape).normal_(memory[CONSTANT], memory[CONSTANT])  # GAUSSIAN VECTOR",
        }
        self.cache = {}

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
                line = "    " + template.format(
                    out=genome.output_gene[i], 
                    in1=genome.input_gene[i], 
                    in2=genome.input_gene_2[i],
                    constant=genome.constants_gene[i],
                    constant_2=genome.constants_gene_2[i]
                )
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
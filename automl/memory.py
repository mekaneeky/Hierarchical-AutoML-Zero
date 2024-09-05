import numpy as np
import torch 
import random


# class CentralMemory:
#     def __init__(self, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
#         self.num_scalars = num_scalars
#         self.num_vectors = num_vectors
#         self.num_tensors = num_tensors
#         self.scalar_size = scalar_size
#         self.vector_size = vector_size
#         self.tensor_size = tensor_size
        
#         self.initialize_memories()

#     def initialize_memories(self):
#         self.scalar_memory = [torch.zeros(self.scalar_size) for _ in range(self.num_scalars)]
#         self.vector_memory = [torch.zeros(self.vector_size) for _ in range(self.num_vectors)]
#         self.tensor_memory = [torch.zeros(self.tensor_size) for _ in range(self.num_tensors)]

#     def reset(self):
#         self.initialize_memories()


class CentralMemory:
    def __init__(self, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
        
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.num_tensors = num_tensors
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.tensor_size = tensor_size

        self.scalar_memory = [torch.zeros((1,)) for _ in range(num_scalars)] if num_scalars > 0 else []
        self.vector_memory = [torch.zeros(vector_size) for _ in range(num_vectors)] if num_vectors > 0 else []
        self.tensor_memory = [torch.zeros(tensor_size) for _ in range(num_tensors)] if num_tensors > 0 else []
        #breakpoint()

        self.scalar_range = range(num_scalars)
        self.vector_range = range(num_scalars, num_scalars + num_vectors)
        self.tensor_range = range(num_scalars + num_vectors, num_scalars + num_vectors + num_tensors)

        self.memory_types = []
        if num_scalars > 0:
            self.memory_types.append('scalar')
        if num_vectors > 0:
            self.memory_types.append('vector')
        if num_tensors > 0:
            self.memory_types.append('tensor')

        self._total_len = sum(len(getattr(self, f"{t}_memory")) for t in self.memory_types)

    def initialize_memories(self):
        self.scalar_memory = [torch.zeros(self.scalar_size) for _ in range(self.num_scalars)]
        self.vector_memory = [torch.zeros(self.vector_size) for _ in range(self.num_vectors)]
        self.tensor_memory = [torch.zeros(self.tensor_size) for _ in range(self.num_tensors)]

    def reset(self):
        self.initialize_memories()

    def __getitem__(self, index):
        if index < 0:
            index = self._total_len + index

        if index >= self._total_len:
            raise IndexError(f"Memory index {index} out of range.")

        if index < len(self.scalar_memory):
            return self.scalar_memory[index] 

        index -= len(self.scalar_memory)

        if index < len(self.vector_memory):
            return self.vector_memory[index]
        index -= len(self.vector_memory)

        if index < len(self.tensor_memory):
            return self.tensor_memory[index]

    def __setitem__(self, index, data):
        if index < 0:
            index = self._total_len + index

        if index >= self._total_len:
            raise IndexError(f"Memory index {index} out of range.")

        #for memory_type in self.memory_types:
            
        if index < len(self.scalar_memory):
            self.scalar_memory[index] = data
            return
        index -= len(self.scalar_memory)

        if index < len(self.vector_memory):
            self.vector_memory[index] = data
            return
        index -= len(self.vector_memory)

        if index < len(self.tensor_memory):
            self.tensor_memory[index] = data
            return
        

    def __len__(self):
        return self._total_len
    
    def get_scalar_address(self):
        return random.choice(self.scalar_range)

    def get_vector_address(self):
        return random.choice(self.vector_range)

    def get_tensor_address(self):
        return random.choice(self.tensor_range)

    def is_scalar_address(self, address):
        return address in self.scalar_range

    def is_vector_address(self, address):
        return address in self.vector_range

    def is_tensor_address(self, address):
        return address in self.tensor_range

class HierarchicalMemoryArrays:
    def __init__(self, num_levels, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
        self.memory_levels = [
            MemoryArrays(num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size)
            for _ in range(num_levels)
        ]

    def __getitem__(self, level):
        return self.memory_levels[level]

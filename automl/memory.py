import numpy as np
import torch 

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
        return self._total_len

class HierarchicalMemoryArrays:
    def __init__(self, num_levels, num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size):
        self.memory_levels = [
            MemoryArrays(num_scalars, num_vectors, num_tensors, scalar_size, vector_size, tensor_size)
            for _ in range(num_levels)
        ]

    def __getitem__(self, level):
        return self.memory_levels[level]

import numpy as np
import torch

class FunctionDecoder:
    def __init__(self):
        self.decoding_map = {
            0: self.do_nothing,
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
            13: self.sigmoid,
            14: self.leaky_relu,
            15: self.relu,
            16: self.stable_softmax,
            17: self.mean_axis,
            18: self.std_axis,
            19: self.set_constant_scalar,
            20: self.log_scalar,
            21: self.power_scalar,
            22: self.sqrt_scalar,
            23: self.max_scalar,
            24: self.min_scalar,
            25: self.mod_scalar,
            26: self.sign_scalar,
            27: self.floor_scalar,
            28: self.ceil_scalar,
            29: self.round_scalar,
            30: self.hypot_scalar,
        }

    @staticmethod
    def do_nothing(*args):
        return 0

    @staticmethod
    def identity(*args):
        return args[0]

    @staticmethod
    def add_scalar(*args):
        return torch.add(args[0], args[1])

    @staticmethod
    def sub_scalar(*args):
        return torch.subtract(args[0], args[1])

    @staticmethod
    def multiply_scalar(*args):
        return torch.multiply(args[0], args[1])

    @staticmethod
    def divide_scalar(*args):
        return torch.divide(args[0], args[1])

    @staticmethod
    def abs_scalar(*args):
        return torch.abs(args[0])

    @staticmethod
    def reciprocal_scalar(*args):
        return torch.reciprocal(args[0])

    @staticmethod
    def sin_scalar(*args):
        return torch.sin(args[0])

    @staticmethod
    def cos_scalar(*args):
        return torch.cos(args[0])

    @staticmethod
    def tan_scalar(*args):
        return torch.tan(args[0])

    @staticmethod
    def arcsin_scalar(*args):
        return torch.arcsin(args[0])

    @staticmethod
    def arccos_scalar(*args):
        return torch.arccos(args[0])

    @staticmethod
    def arctan_scalar(*args):
        return torch.arctan(args[0])

    @staticmethod
    def sigmoid(*args):
        return 1 / (1 + torch.exp(-args[0]))

    @staticmethod
    def leaky_relu(*args, alpha=0.01):
        return torch.where(args[0] > 0, args[0], args[0] * alpha)

    @staticmethod
    def relu(*args):
        x = args[0]
        x[x < 0] = 0
        return x

    @staticmethod
    def stable_softmax(*args):
        try:
            x = args[0]
            z = x - torch.max(x)
            numerator = torch.exp(z)
            denominator = torch.sum(numerator)
            softmax = numerator / denominator
            return softmax
        except:
            return x

    @staticmethod
    def mean_axis(*args):
        x = args[0]
        try:
            return torch.mean(x, axis=0)
        except:
            return x

    @staticmethod
    def std_axis(*args):
        x = args[0]
        try:
            return torch.std(x, axis=0)
        except:
            return x

    @staticmethod
    def set_constant_scalar(c, _):
        return c

    @staticmethod
    def log_scalar(*args):
        return torch.log(torch.abs(args[0]) + 1e-10)

    @staticmethod
    def power_scalar(*args):
        return torch.pow(args[0], args[1])

    @staticmethod
    def sqrt_scalar(*args):
        return torch.sqrt(torch.abs(args[0]))

    @staticmethod
    def max_scalar(*args):
        return torch.maximum(args[0], args[1])

    @staticmethod
    def min_scalar(*args):
        return torch.minimum(args[0], args[1])

    @staticmethod
    def mod_scalar(*args):
        return torch.mod(args[0], args[1])

    @staticmethod
    def sign_scalar(*args):
        return torch.sign(args[0])

    @staticmethod
    def floor_scalar(*args):
        return torch.floor(args[0])

    @staticmethod
    def ceil_scalar(*args):
        return torch.ceil(args[0])

    @staticmethod
    def round_scalar(*args):
        return torch.round(args[0])

    @staticmethod
    def hypot_scalar(*args):
        return torch.hypot(args[0], args[1])

    def decode(self, genome):
        decoded_functions = []
        for op in genome.gene:
            if op in self.decoding_map:
                decoded_functions.append(self.decoding_map[op])
        return decoded_functions, genome

class NumpyFunctionDecoder:
    def __init__(self):
        self.decoding_map = {
            0: self.do_nothing,
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
            13: self.sigmoid,
            14: self.leaky_relu,
            15: self.relu,
            16: self.stable_softmax,
            17: self.mean_axis,
            18: self.std_axis,
            19: self.set_constant_scalar,
            20: self.log_scalar,
            21: self.power_scalar,
            22: self.sqrt_scalar,
            23: self.max_scalar,
            24: self.min_scalar,
            25: self.mod_scalar,
            26: self.sign_scalar,
            27: self.floor_scalar,
            28: self.ceil_scalar,
            29: self.round_scalar,
            30: self.hypot_scalar,
        }

    @staticmethod
    def do_nothing(*args):
        return 0

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
    def sigmoid(*args):
        return 1 / (1 + np.exp(-args[0]))

    @staticmethod
    def leaky_relu(*args, alpha=0.01):
        return np.where(args[0] > 0, args[0], args[0] * alpha)

    @staticmethod
    def relu(*args):
        x = args[0]
        x[x < 0] = 0
        return x

    @staticmethod
    def stable_softmax(*args):
        try:
            x = args[0]
            z = x - np.max(x)
            numerator = np.exp(z)
            denominator = np.sum(numerator)
            softmax = numerator / denominator
            return softmax
        except:
            return x

    @staticmethod
    def mean_axis(*args):
        x = args[0]
        try:
            return np.mean(x, axis=0)
        except:
            return x

    @staticmethod
    def std_axis(*args):
        x = args[0]
        try:
            return np.std(x, axis=0)
        except:
            return x

    @staticmethod
    def set_constant_scalar(c, _):
        return c

    @staticmethod
    def log_scalar(*args):
        return np.log(np.abs(args[0]) + 1e-10)

    @staticmethod
    def power_scalar(*args):
        return np.power(args[0], args[1])

    @staticmethod
    def sqrt_scalar(*args):
        return np.sqrt(np.abs(args[0]))

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

    def decode(self, genome):
        decoded_functions = []
        for op in genome.gene:
            if op in self.decoding_map:
                decoded_functions.append(self.decoding_map[op])
        return decoded_functions, genome

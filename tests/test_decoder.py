import numpy as np
import torch
from automl.function_decoder import FunctionDecoder
from automl.memory import MemoryArrays

def generate_test_data(show_failed_only=True):
    # Initialize the function decoder
    decoder = FunctionDecoder()

    # Initialize memory with appropriate sizes
    memory = MemoryArrays(num_scalars=10, num_vectors=10, num_tensors=10, scalar_size=1, vector_size=10, tensor_size=(10, 10))

    # Test each function in the decoder
    success_count = 0
    failure_count = 0
    for op, (func, output_type, input_1_type, input_2_type) in decoder.decoding_map.items():
               # Generate inputs based on the expected types
        if input_1_type == "scalar":
            input1 = memory[memory.get_scalar_address()]
        elif input_1_type == "vector":
            input1 = memory[memory.get_vector_address()]
        elif input_1_type == "matrix":
            input1 = memory[memory.get_tensor_address()]
        else:
            input1 = None

        if input_2_type == "scalar":
            input2 = memory[memory.get_scalar_address()]
        elif input_2_type == "vector":
            input2 = memory[memory.get_vector_address()]
        elif input_2_type == "matrix":
            input2 = memory[memory.get_tensor_address()]
        else:
            input2 = None

        try:
            # Call the function with the generated inputs
            output = func(input1, input2, 0.5, 0.5, 10, 10)
            if output_type == "scalar" and not output.squeeze().ndim==0:
                raise TypeError(f"Expected output type 'scalar', but got {output.shape}")
            elif output_type == "vector" and not output.ndim==1:
                raise TypeError(f"Expected output type 'vector', but got {output.shape}")
            elif output_type == "matrix" and not output.ndim==2:
                raise TypeError(f"Expected output type 'matrix', but got {output.shape}")

            if not show_failed_only:
                print(f"Testing function {func.__name__} (op code: {op})")
                print(f"Output: {output}\n")
            success_count += 1
        except Exception as e:
            #breakpoint()
            print(f"Testing function {func.__name__} (op code: {op})")
            print(f"Function {func.__name__} failed with error: {e}")
            print(f"Output: {output}\n")
            failure_count += 1

    print(f"Total functions executed successfully: {success_count}")
    print(f"Total functions failed: {failure_count}/{(failure_count+success_count)}")

if __name__ == "__main__":
    generate_test_data()
import json
import torch
from automl.genome import FunctionGenome
from automl.memory import CentralMemory
from automl.function_decoder import FunctionDecoder

def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def list_to_tensor(obj):
    if isinstance(obj, list):
        return torch.tensor(obj)
    elif isinstance(obj, dict):
        return {k: list_to_tensor(v) for k, v in obj.items()}
    else:
        return obj

def export_gene_to_json(gene, filename = None):
    """
    Export a FunctionGenome object to a JSON file.
    
    Args:
    gene (FunctionGenome): The gene to export
    filename (str): The name of the file to save the JSON data
    """

    gene_data = {
        "length": gene.length,
        "meta_level": gene.meta_level,
        "gene": gene.gene,
        "input_gene": gene.input_gene,
        "input_gene_2": gene.input_gene_2,
        "output_gene": gene.output_gene,
        "constants_gene": gene.constants_gene,
        "constants_gene_2": gene.constants_gene_2,
        "row_fixed": gene.row_fixed,
        "column_fixed": gene.column_fixed,
        "version": gene.version,
        "fitness": gene.fitness,
        "memory": {
            "num_scalars": gene.memory.num_scalars,
            "num_vectors": gene.memory.num_vectors,
            "num_tensors": gene.memory.num_tensors,
            "scalar_size": tensor_to_list(gene.memory.scalar_size),
            "vector_size": tensor_to_list(gene.memory.vector_size),
            "tensor_size": tensor_to_list(gene.memory.tensor_size)
        }
    }
    
    if filename is None:
        return gene_data
    
    with open(filename, 'w') as f:
        json.dump(gene_data, f, indent=2)

def calculate_total_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size    

def validate_gene_size(gene_data, config):
    """
    Validate the size of gene data against configuration limits.
    
    Args:
    gene_data (dict): The gene data to validate
    config (object): Configuration object containing size limits
    
    Raises:
    ValueError: If any size limit is exceeded
    """
    checks = [
        ('memory.num_scalars', lambda: gene_data['memory']['num_scalars'], config.max_num_scalars),
        ('memory.num_vectors', lambda: gene_data['memory']['num_vectors'], config.max_num_vectors),
        ('memory.num_tensors', lambda: gene_data['memory']['num_tensors'], config.max_num_tensors),
        ('memory.vector_size', lambda: calculate_total_size(gene_data['memory']['vector_size']), config.max_vector_size),
        ('memory.tensor_size', lambda: calculate_total_size(gene_data['memory']['tensor_size']), config.max_tensor_size),
        ('length', lambda: gene_data['length'], config.max_gene_length),
        ('input_gene', lambda: len(gene_data['input_gene']), config.max_gene_length),
        ('input_gene_2', lambda: len(gene_data['input_gene_2']), config.max_gene_length),
        ('output_gene', lambda: len(gene_data['output_gene']), config.max_gene_length),
        ('constants_gene', lambda: len(gene_data['constants_gene']), config.max_gene_length),
        ('constants_gene_2', lambda: len(gene_data['constants_gene_2']), config.max_gene_length),
        #('row_fixed', lambda: len(gene_data['row_fixed']), config.max_gene_length),
        #('column_fixed', lambda: len(gene_data['column_fixed']), config.max_gene_length),
    ]
    
    for name, size_func, limit in checks:
        if size_func() > limit:
            raise ValueError(f"Gene size exceeded: {name}")

def import_gene_from_json(gene_data = None, filename = None, function_decoder=None, config= None):
    """
    Import a FunctionGenome object from a JSON file.
    
    Args:
    filename (str): The name of the file to load the JSON data from
    function_decoder (FunctionDecoder): The function decoder to use for the imported gene
    
    Returns:
    FunctionGenome: The imported gene
    """
    if gene_data is None:
        with open(filename, 'r') as f:
            gene_data = json.load(f)

    if config is not None:
        validate_gene_size(gene_data, config)

    # Recreate the CentralMemory object
    memory = CentralMemory(
        num_scalars=gene_data['memory']['num_scalars'],
        num_vectors=gene_data['memory']['num_vectors'],
        num_tensors=gene_data['memory']['num_tensors'],
        scalar_size=gene_data['memory']['scalar_size'],
        vector_size=tuple(gene_data['memory']['vector_size']),
        tensor_size=tuple(gene_data['memory']['tensor_size'])
    )
    
    # Create a new FunctionGenome 
    if function_decoder:
        gene = FunctionGenome(
            length=gene_data['length'],
            central_memory=memory,
            function_decoder=function_decoder,
            meta_level=gene_data['meta_level']
        )
    else:
        gene = FunctionGenome(
            length=gene_data['length'],
            central_memory=memory,
            function_decoder=FunctionDecoder(),
            meta_level=gene_data['meta_level']
        )
    
    # Populate the gene with the imported data
    gene.gene = gene_data['gene']
    gene.input_gene = gene_data['input_gene']
    gene.input_gene_2 = gene_data['input_gene_2']
    gene.output_gene = gene_data['output_gene']
    gene.constants_gene = gene_data['constants_gene']
    gene.constants_gene_2 = gene_data['constants_gene_2']
    gene.row_fixed = gene_data['row_fixed']
    gene.column_fixed = gene_data['column_fixed']
    gene.version = gene_data['version']
    gene.fitness = None#gene_data['fitness']
    
    return gene



# Example usage:
if __name__ == "__main__":
    # Assuming you have a gene and function_decoder object
    # gene = ...
    # function_decoder = FunctionDecoder()
    
    # Export gene to JSON
    # export_gene_to_json(gene, "gene_data.json")
    
    # Import gene from JSON
    # imported_gene = import_gene_from_json("gene_data.json", function_decoder)
    pass
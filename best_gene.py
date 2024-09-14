from automl.gene_io import import_gene_from_json
from automl.unpacker import HierarchicalGenomeUnpacker
from automl.function_decoder import FunctionDecoder

decoder = FunctionDecoder()
best_gene = import_gene_from_json("best_gene.json",decoder)
unpacker = HierarchicalGenomeUnpacker()
print(unpacker.unpack_function_genome(best_gene))
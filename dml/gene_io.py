import json
from deap import gp, creator

def save_individual_to_json(individual):
    expr_str = str(individual)
    return json.dumps({'expression': expr_str})

def load_individual_from_json(data=None, pset=None, toolbox=None, filename = None):
    if filename is not None:
        with open(filename, "r") as fd:
            data = json.loads(fd.read())
        if type(data) == str:
            data = json.loads(data)
        
    expr_str = data['expression']
    expr = gp.PrimitiveTree.from_string(expr_str, pset)
    individual = creator.Individual(expr)
    func = toolbox.compile(expr=individual)
    return individual, func
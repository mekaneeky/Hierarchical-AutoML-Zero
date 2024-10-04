import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from deap import base, creator, tools, gp
import operator
import random
import math
import numpy as np
import pickle
import os
import logging
import traceback
from functools import partial

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Local storage path
LOCAL_STORAGE_PATH = "./best_individuals"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom function for safe division
# def safe_div(x, y):
#     return torch.where(y != 0, x / y, torch.zeros_like(x))
device = torch.device("mps") 
device_str = str(device)

def safe_div(x, y):
    epsilon = 1e-8
    return x / (y + epsilon)

def safe_add(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x + y

def safe_sub(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x - y

def safe_mul(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x * y

def safe_div(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    epsilon = 1e-8
    return x / (y + epsilon)

def safe_sigmoid(x):
    x = torch.tensor(x, device=device,dtype=torch.float32) if not torch.is_tensor(x) else x
    return torch.sigmoid(x)

def safe_relu(x):
    x = torch.tensor(x, device=device,dtype=torch.float32) if not torch.is_tensor(x) else x
    return torch.relu(x)

def safe_tanh(x):
    x = torch.tensor(x, device=device,dtype=torch.float32) if not torch.is_tensor(x) else x
    return torch.tanh(x)

def rand_const():
    return torch.tensor(random.uniform(-10, 10))

def random_uniform_tensor(low, high, device='cpu'):
    return torch.tensor(random.uniform(low, high), device=device)
generate_random_tensor_partial = partial(random_uniform_tensor, -1, 1, device=device_str)


# # Simplified primitive set
# pset = gp.PrimitiveSet("MAIN", 2)
# pset.addPrimitive(torch.add, 2)
# pset.addPrimitive(torch.mul, 2)
# pset.addPrimitive(torch.sub, 2)
# pset.addPrimitive(safe_div, 2)
# pset.addTerminal(1.0)
# pset.addTerminal(0.1)
# pset.renameArguments(ARG0='y_pred')
# pset.renameArguments(ARG1='y_true')
# # Adding activation functions to the primitive set
# pset.addPrimitive(torch.sigmoid, 1)
# pset.addPrimitive(torch.tanh, 1)
# pset.addPrimitive(torch.relu, 1)


# Update the primitive set after defining device
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(safe_add, 2)
pset.addPrimitive(safe_sub, 2)
pset.addPrimitive(safe_mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_sigmoid, 1)
pset.addPrimitive(safe_relu, 1)
pset.addPrimitive(safe_tanh, 1)
# pset.addEphemeralConstant(name='const', ephemeral=generate_random_tensor_partial)
pset.renameArguments(ARG0='y_pred')
pset.renameArguments(ARG1='y_true')
pset.addTerminal(torch.tensor(1.0, device=device), name='one')
pset.addTerminal(torch.tensor(0.1, device=device), name='zero_point_one')



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def safe_evaluate(func, outputs, labels):
    try:
        loss = func(outputs, labels)
        
        if loss is None:
            logging.error(f"Loss function returned None: {func}")
            return torch.tensor(float('inf'), device=outputs.device)
        
        if not torch.is_tensor(loss):
            logging.error(f"Loss function didn't return a tensor: {type(loss)}")
            return torch.tensor(float('inf'), device=outputs.device)
        
        if not torch.isfinite(loss).all():
            logging.warning(f"Non-finite loss detected: {loss}")
            return torch.tensor(float('inf'), device=outputs.device)
        
        if loss.ndim > 0:
            loss = loss.mean()
        
        return loss
    except Exception as e:
        logging.error(f"Error in loss calculation: {str(e)}")
        logging.error(traceback.format_exc())
        return torch.tensor(float('inf'), device=outputs.device)



def eval_loss_function(individual, trainloader, device):
    func = toolbox.compile(expr=individual)
    logging.debug(f"Evaluating individual: {str(individual)}")
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    total_loss = 0.0
    num_batches = 0
    
    try:
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 10:  # Limit to 10 batches for quick evaluation
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Add this line
            outputs = model(inputs)
            
            # Convert labels to one-hot encoding
            labels_one_hot = torch.zeros(labels.size(0), 10, device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            loss = safe_evaluate(func, outputs, labels_one_hot)
            if torch.isinf(loss):
                logging.warning(f"Infinite loss for individual: {str(individual)}")
                return float('inf'),
    
            loss.backward()  # Add this line
            optimizer.step()  # Add this line

            total_loss += loss.item()
            num_batches += 1
    
        average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logging.debug(f"Individual evaluation - Average Loss: {average_loss:.4f}")
        return average_loss,
    except Exception as e:
        logging.error(f"Error during individual evaluation: {str(e)}")
        logging.error(traceback.format_exc())
        return float('inf'),


toolbox.register("evaluate", eval_loss_function)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    logging.info("Starting evolutionary process")

    for gen in range(10):  # Reduced to 10 generations for quicker debugging
        logging.info(f"Generation {gen}")
        
        # Evaluate the entire population
        for i, ind in enumerate(pop):
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, trainloader, device)
            logging.debug(f"Gen {gen}, Individual {i}: Fitness = {ind.fitness.values[0]}")

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for i, ind in enumerate(invalid_ind):
            ind.fitness.values = toolbox.evaluate(ind, trainloader, device)
            logging.debug(f"Gen {gen}, New Individual {i}: Fitness = {ind.fitness.values[0]}")

        # Update the hall of fame with the generated individuals
        hof.update(pop)

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop if not math.isinf(ind.fitness.values[0])]
        
        length = len(pop)
        mean = sum(fits) / length if fits else float('inf')
        sum2 = sum(x*x for x in fits) if fits else float('inf')
        std = abs(sum2 / length - mean**2)**0.5 if fits else float('inf')
        
        logging.info(f"Generation {gen}: Valid fits: {len(fits)}/{length}")
        logging.info(f"Min {min(fits) if fits else float('inf')}")
        logging.info(f"Max {max(fits) if fits else float('inf')}")
        logging.info(f"Avg {mean}")
        logging.info(f"Std {std}")

        # Save the best individual
        best = tools.selBest(pop, 1)[0]
        with open(os.path.join(LOCAL_STORAGE_PATH, f'best_individual_gen_{gen}.pkl'), 'wb') as f:
            pickle.dump(best, f)

    logging.info("Evolution finished")

    best = hof[0]
    logging.info(f"Best individual: {str(best)}")
    logging.info(f"Best fitness: {best.fitness.values[0]}")

if __name__ == "__main__":
    main()
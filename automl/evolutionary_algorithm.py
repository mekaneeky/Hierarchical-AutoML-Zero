import random
import numpy as np
from .genome import FunctionGenome, HierarchicalGenome

class AutoMLZero:
    def __init__(self, population_size, num_meta_levels, genome_length, tournament_size,
                 central_memory, function_decoder):
        self.population_size = population_size
        self.num_meta_levels = num_meta_levels
        self.genome_length = genome_length
        self.tournament_size = tournament_size
        self.central_memory = central_memory
        self.hierarchical_genome = HierarchicalGenome(num_meta_levels, genome_length, central_memory, function_decoder, population_size)
        self.function_decoder = function_decoder
        
    def mutate(self, genome, level):
        new_genome = FunctionGenome(
            self.genome_length,
            self.central_memory,
            self.function_decoder,
            meta_level=level,
            lower_level_population=self.hierarchical_genome.genomes[level - 1] if level > 0 else None
        )
        new_genome.gene = genome.gene.copy()
        new_genome.input_gene = genome.input_gene.copy()
        new_genome.input_gene_2 = genome.input_gene_2.copy()
        new_genome.output_gene = genome.output_gene.copy()
        new_genome.mutate()
        return new_genome
    
    def evaluate_population(self, population):
        for genome in population:
            self.get_fitness(genome)

    def get_fitness(self, genome):
        return genome.fitness

    def genome_to_key(self, genome):
        # Include version numbers of lower-level genomes in the key
        lower_level_versions = tuple(g.version for g in genome.lower_level_population) if genome.lower_level_population else ()
        return (genome.version, tuple(genome.gene), tuple(genome.input_gene),
                tuple(genome.input_gene_2), tuple(genome.output_gene), lower_level_versions)

    def tournament_selection(self, population):
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda genome: self.get_fitness(genome))

    def select_best_genomes(self, population, num_best=10):
        return sorted(population, key=lambda genome: self.get_fitness(genome), reverse=True)[:num_best]


# DEPRECATED 
# class LevelWiseEvolutionaryAlgorithm:
#     def __init__(self, population_size, num_meta_levels, genome_length, tournament_size, mutation_probability,
#                  hierarchical_memory, function_decoder, input_data):
#         self.population_size = population_size
#         self.num_meta_levels = num_meta_levels
#         self.genome_length = genome_length
#         self.tournament_size = tournament_size
#         self.mutation_probability = mutation_probability
#         self.hierarchical_memory = hierarchical_memory
#         self.function_decoder = function_decoder
#         self.input_data = input_data
#         self.hierarchical_genome = HierarchicalGenome(num_meta_levels, genome_length, hierarchical_memory, function_decoder, population_size)
#         self.fitness_cache = {}  # Cache to store fitness scores and evaluation flags

#     def evolve(self, fitness_evaluator, num_generations, meta_epochs):
#         for meta_epoch in range(meta_epochs):
#             for level in range(self.num_meta_levels):
#                 print(f"Evolving Level {level}")
#                 population = self.hierarchical_genome.genomes[level]
#                 self.evaluate_level(level, population, fitness_evaluator)

#                 for generation in range(num_generations[level]):
#                     # Tournament selection
#                     parent = self.tournament_selection(population, fitness_evaluator)

#                     # Create and mutate offspring
#                     offspring = self.mutate(parent, level)

#                     # Add offspring to population and remove oldest member
#                     population.append(offspring)
#                     population.pop(0)

#                     # Mark offspring for re-evaluation
#                     self.fitness_cache[self.genome_to_key(offspring)] = (None, True)

#                     # Update the level's population in the hierarchical genome
#                     self.hierarchical_genome.genomes[level] = population

#                     # Calculate and print best fitness
#                     best_fitness = max(self.get_fitness(genome, fitness_evaluator) for genome in population)
#                     print(f"Meta-Epoch {meta_epoch + 1} Generation {generation + 1}, Level {level}: Best Fitness = {best_fitness:.4f}")

#                 # Update lower_level_population for the next level with the whole population
#                 if level < self.num_meta_levels - 1:
#                     for genome in self.hierarchical_genome.genomes[level + 1]:
#                         genome.lower_level_population = population
#                         # Invalidate cache for higher level genomes
#                         self.fitness_cache[self.genome_to_key(genome)] = (None, True)

#         return self.select_best_genomes(self.hierarchical_genome.genomes[-1], fitness_evaluator, num_best=1)[0]


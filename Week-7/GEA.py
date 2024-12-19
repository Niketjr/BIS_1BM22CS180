import numpy as np
import random

def objective_function(x):
    return x**2

population_size = 20
num_genes = 1  
mutation_rate = 0.1
crossover_rate = 0.8
num_generations = 100
bounds = (-10, 10)  

def initialize_population(size, bounds, num_genes):
    return np.random.uniform(bounds[0], bounds[1], (size, num_genes))

def evaluate_fitness(population):
    return np.array([objective_function(ind[0]) for ind in population])

def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(range(len(population)), tournament_size, replace=False)
        best = indices[np.argmin(fitness[indices])]
        selected.append(population[best])
    return np.array(selected)

def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[min(i + 1, len(parents) - 1)]
        if len(parent1) > 1 and random.random() < crossover_rate: 
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:  
            child1, child2 = parent1.copy(), parent2.copy()
        offspring.extend([child1, child2])
    return np.array(offspring[:len(parents)])

def mutate(population, mutation_rate, bounds):
    for individual in population:
        if random.random() < mutation_rate:
            mutation_value = np.random.normal(0, 1) 
            individual[0] = np.clip(individual[0] + mutation_value, bounds[0], bounds[1])
    return population

population = initialize_population(population_size, bounds, num_genes)
best_solution = None
best_fitness = float('inf')

for generation in range(num_generations):
    fitness = evaluate_fitness(population)
    current_best_idx = np.argmin(fitness)
    if fitness[current_best_idx] < best_fitness:
        best_fitness = fitness[current_best_idx]
        best_solution = population[current_best_idx]
    
    selected_population = tournament_selection(population, fitness)
    offspring_population = crossover(selected_population, crossover_rate)
    population = mutate(offspring_population, mutation_rate, bounds)
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

print(f"Best solution: x = {best_solution[0]}, f(x) = {best_fitness}")

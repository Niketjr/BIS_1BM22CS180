import random
import numpy as np

def genetic_algorithm(
    fitness_function, 
    solution_bounds, 
    population_size=100, 
    crossover_rate=0.8, 
    mutation_rate=0.1, 
    generations=50
):
    """
    Genetic Algorithm for optimization.

    Parameters:
        fitness_function: callable
            Function to evaluate solution quality.
        solution_bounds: tuple
            Bounds for each dimension of the solution (lower, upper).
        population_size: int
            Number of individuals in the population.
        crossover_rate: float
            Probability of crossover.
        mutation_rate: float
            Probability of mutation.
        generations: int
            Number of generations.

    Returns:
        Best solution and its fitness.
    """
    lower_bound, upper_bound = solution_bounds
    num_dimensions = len(lower_bound)

    # Initialize population
    population = np.random.uniform(lower_bound, upper_bound, (population_size, num_dimensions))

    def select_parents(population, fitness):
        """Select two parents using tournament selection."""
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        return population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]

    def crossover(parent1, parent2):
        """Perform single-point crossover."""
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, num_dimensions - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(individual):
        """Apply mutation to an individual."""
        for i in range(num_dimensions):
            if random.random() < mutation_rate:
                individual[i] = random.uniform(lower_bound[i], upper_bound[i])

    # Evaluate initial population
    fitness = np.array([fitness_function(ind) for ind in population])

    for generation in range(generations):
        new_population = []

        # Elitism: retain the best individual
        best_index = np.argmax(fitness)
        new_population.append(population[best_index])

        # Generate new population
        while len(new_population) < population_size:
            parent1 = select_parents(population, fitness)
            parent2 = select_parents(population, fitness)

            child1, child2 = crossover(parent1, parent2)

            mutate(child1)
            mutate(child2)

            new_population.extend([child1, child2])

        population = np.array(new_population[:population_size])
        fitness = np.array([fitness_function(ind) for ind in population])

        # Log progress
        best_fitness = np.max(fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Return the best solution
    best_index = np.argmax(fitness)
    return population[best_index], fitness[best_index]

# Example usage
def example_fitness_function(x):
    """Example fitness function: Sphere function."""
    return -np.sum(x**2)  # Minimize sum of squares (negative for maximization)

# Define problem bounds (2D problem)
solution_bounds = ([-10, -10], [10, 10])

best_solution, best_fitness = genetic_algorithm(
    fitness_function=example_fitness_function,
    solution_bounds=solution_bounds,
    population_size=50,
    generations=20
)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
print("Niket Dugar 1BM22CS180")

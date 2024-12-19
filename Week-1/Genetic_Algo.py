import random

# Genetic Algorithm parameters
POPULATION_SIZE = 100
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP 
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
TARGET = "cogito ergo sum"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def mutated_gene(cls):
        return random.choice(GENES)

    @classmethod
    def create_genome(cls):
        return [cls.mutated_gene() for _ in range(len(TARGET))]

    def mate(self, partner):
        """
        Perform crossover and mutation to generate offspring.
        """
        child_chromosome = [
            gp1 if random.random() < 0.45 else
            gp2 if random.random() < 0.90 else
            self.mutated_gene()
            for gp1, gp2 in zip(self.chromosome, partner.chromosome)
        ]
        return Individual(child_chromosome)

    def calculate_fitness(self):
        """
        Calculate fitness as the number of characters that differ from the target.
        """
        return sum(gs != gt for gs, gt in zip(self.chromosome, TARGET))

def main():
    generation = 1
    population = [Individual(Individual.create_genome()) for _ in range(POPULATION_SIZE)]

    while True:
        # Sort population by fitness (lower is better)
        population.sort(key=lambda x: x.fitness)

        # Check if the best individual has reached the target
        if population[0].fitness == 0:
            break

        # Elitism: carry over top 10% of the population
        next_generation = population[:POPULATION_SIZE // 10]

        # Generate the remaining 90% of the new population
        for _ in range(POPULATION_SIZE - len(next_generation)):
            parent1 = random.choice(population[:50])  # Top 50 for parent selection
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            next_generation.append(child)

        population = next_generation

        # Print the best individual of this generation
        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1

    # Print the final result
    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == '__main__':
    print("Prajwal.P 1BM22CS200\n")
    main()

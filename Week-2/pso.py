import random
import math
import copy
import sys

# Fitness functions
def fitness_rastrigin(position):
    return sum((xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10 for xi in position)

def fitness_sphere(position):
    return sum(xi * xi for xi in position)

# Particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        rnd = random.Random(seed)
        self.position = [rnd.uniform(minx, maxx) for _ in range(dim)]
        self.velocity = [rnd.uniform(minx, maxx) for _ in range(dim)]
        self.fitness = fitness(self.position)
        self.best_part_pos = self.position[:]
        self.best_part_fitnessVal = self.fitness

# PSO algorithm
def pso(fitness, max_iter, n, dim, minx, maxx):
    w, c1, c2 = 0.729, 1.49445, 1.49445
    rnd = random.Random(0)
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]
    best_swarm_pos = swarm[0].position[:]
    best_swarm_fitnessVal = sys.float_info.max

    for particle in swarm:
        if particle.fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = particle.fitness
            best_swarm_pos = particle.position[:]

    for Iter in range(max_iter):
        if Iter % 10 == 0 and Iter > 0:
            print(f"Iter = {Iter} best fitness = {best_swarm_fitnessVal:.3f}")
        for particle in swarm:
            for k in range(dim):
                r1, r2 = rnd.random(), rnd.random()
                particle.velocity[k] = (
                    w * particle.velocity[k]
                    + c1 * r1 * (particle.best_part_pos[k] - particle.position[k])
                    + c2 * r2 * (best_swarm_pos[k] - particle.position[k])
                )
                particle.velocity[k] = max(minx, min(maxx, particle.velocity[k]))
                particle.position[k] += particle.velocity[k]

            particle.fitness = fitness(particle.position)
            if particle.fitness < particle.best_part_fitnessVal:
                particle.best_part_fitnessVal = particle.fitness
                particle.best_part_pos = particle.position[:]
            if particle.fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = particle.fitness
                best_swarm_pos = particle.position[:]

    return best_swarm_pos

# Driver code
def run_pso(fitness_func, func_name, dim=3, num_particles=50, max_iter=100):
    print(f"\nBegin particle swarm optimization on {func_name} function\n")
    print(f"Goal is to minimize {func_name} function in {dim} variables")
    print(f"Function has known min = 0.0 at ({', '.join(['0'] * dim)})")
    print(f"Setting num_particles = {num_particles}")
    print(f"Setting max_iter = {max_iter}")
    print("\nStarting PSO algorithm\n")

    best_position = pso(fitness_func, max_iter, num_particles, dim, -10.0, 10.0)

    print("\nPSO completed\n")
    print("\nBest solution found:")
    print(["%.6f" % x for x in best_position])
    print(f"Fitness of best solution = {fitness_func(best_position):.6f}")
    print(f"\nEnd particle swarm for {func_name} function\n")

# Run for Rastrigin and Sphere functions
print("Niket Dugar 1BMM2CS180")
run_pso(fitness_rastrigin, "Rastrigin")
run_pso(fitness_sphere, "Sphere")



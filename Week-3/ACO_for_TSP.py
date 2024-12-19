import numpy as np
import random
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    def __init__(self, cities, num_ants=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, iterations=100):
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.heuristic = 1 / (np.linalg.norm(self.cities[:, None] - self.cities[None, :], axis=-1) + np.eye(self.num_cities))

    def _select_next_city(self, current_city, visited):
        probabilities = (self.pheromone[current_city] ** self.alpha) * (self.heuristic[current_city] ** self.beta)
        probabilities[visited] = 0
        return np.random.choice(self.num_cities, p=probabilities / probabilities.sum())

    def _update_pheromones(self, routes, lengths):
        self.pheromone *= (1 - self.evaporation_rate)
        for route, length in zip(routes, lengths):
            for i in range(self.num_cities):
                self.pheromone[route[i], route[i + 1]] += 1 / length

    def solve(self):
        best_route, best_length = None, float('inf')
        history = []

        for iteration in range(self.iterations):
            routes, lengths = [], []

            for _ in range(self.num_ants):
                visited = np.zeros(self.num_cities, dtype=bool)
                current_city = random.randint(0, self.num_cities - 1)
                route = [current_city]
                visited[current_city] = True

                for _ in range(self.num_cities - 1):
                    next_city = self._select_next_city(current_city, visited)
                    route.append(next_city)
                    visited[next_city] = True
                    current_city = next_city

                route.append(route[0])
                length = sum(np.linalg.norm(self.cities[route[i]] - self.cities[route[i + 1]]) for i in range(self.num_cities))
                routes.append(route)
                lengths.append(length)

                if length < best_length:
                    best_length, best_route = length, route

            print(f"Iteration {iteration + 1}: Best Length = {best_length:.2f}")
            history.append((iteration, best_route, best_length))
            self._update_pheromones(routes, lengths)

        return best_route, best_length, history

    def plot_route(self, route, iteration, length):
        plt.figure(figsize=(8, 6))
        x, y = self.cities[route, 0], self.cities[route, 1]
        plt.plot(x, y, 'r-', linewidth=2)
        plt.scatter(self.cities[:, 0], self.cities[:, 1], s=100, c='blue', label='Cities')
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title(f"ACO Route - Iteration {iteration + 1}, Length = {length:.2f}")
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    num_points = 200
    x_range, y_range = (0, 1000), (0, 1000)
    cities = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]

    aco = AntColonyOptimizer(cities)
    best_route, best_length, history = aco.solve()
    
    print("Niket Dugar 1BM22CS180")
    print("Best Route:", best_route)
    print("Best Length:", best_length)

    for i in [0, 24, 49, 99]:
        _, route, length = history[i]
        aco.plot_route(route, i, length)

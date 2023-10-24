import numpy as np
import random
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

city_data = pd.read_csv("/Users/rcross/Desktop/Work/Task/CI/P2/st70.csv")
dist_matrix = distance_matrix(city_data.values, city_data.values)
dist = dist_matrix.tolist()
cities = list(range(70))

def route_distance(route):
    return sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1)) + dist[route[-1]][route[0]]

def crossover(parent1, parent2):
    child = []
    a, b = sorted(random.sample(range(len(parent1)), 2))
    child[a:b] = parent1[a:b]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child

def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


def ga_tsp_viz(dist, pop_size=100, generations=5000, mutation_rate=0.01, elite_size=10):
    population = [list(np.random.permutation(len(dist))) for _ in range(pop_size)]

    # List to store the best route's distance for each generation
    best_distances = []

    for _ in range(generations):
        population = sorted(population, key=route_distance)
        best_distances.append(route_distance(population[0]))

        new_generation = []
        elite = population[:elite_size]
        pool = population[:elite_size * 2]
        for _ in range(pop_size - elite_size):
            parent1, parent2 = random.choices(pool, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_generation.append(child)
        population = elite + new_generation

    best_route = sorted(population, key=route_distance)[0]
    return best_route, route_distance(best_route), best_distances

best_route, best_distance, best_distances = ga_tsp_viz(dist)
city_route = [cities[i] for i in best_route]
print(f"Best route: {' -> '.join(map(str, city_route))}\nTotal distance: {best_distance}")

plt.figure(figsize=(12,6))
plt.plot(best_distances)
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.title('Convergence of the Genetic Algorithm')
plt.grid(True)
plt.savefig('/Users/rcross/Desktop/Work/Task/CI/P2/GA.png')

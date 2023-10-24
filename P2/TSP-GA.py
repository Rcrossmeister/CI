import numpy as np
import random

dist = [
    [0, 8, 3, 1, 4, 9, 3, 6],
    [8, 0, 5, 10, 11, 4, 3, 6],
    [3, 5, 0, 8, 7, 1, 5, 12],
    [1, 10, 8, 0, 9, 11, 6, 4],
    [4, 11, 7, 9, 0, 5, 17, 3],
    [9, 4, 1, 11, 5, 0, 4, 1],
    [3, 3, 5, 6, 17, 4, 0, 7],
    [6, 6, 12, 4, 3, 1, 7, 0]
]

cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

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

def ga_tsp(dist, pop_size=100, generations=500, mutation_rate=0.01, elite_size=10):
    population = [list(np.random.permutation(len(dist))) for _ in range(pop_size)]
    for _ in range(generations):
        population = sorted(population, key=route_distance)
        new_generation = []
        elite = population[:elite_size]
        pool = population[:elite_size*2]
        for _ in range(pop_size - elite_size):
            parent1, parent2 = random.choices(pool, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_generation.append(child)
        population = elite + new_generation
    return sorted(population, key=route_distance)[0], route_distance(sorted(population, key=route_distance)[0])

best_route, best_distance = ga_tsp(dist)
city_route = [cities[i] for i in best_route]
print(f"Best route: {' -> '.join(city_route)}\nTotal distance: {best_distance}")

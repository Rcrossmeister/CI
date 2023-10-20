import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

# Define the cities and their coordinates
cities = {
    'A': (0, 0),
    'B': (1, 2),
    'C': (2, 4),
    'D': (3, 6),
    'E': (4, 1),
    'F': (5, 3),
    'G': (6, 5),
    'H': (7, 0),
    'I': (8, 2),
    'J': (9, 4)
}

# Parameters for the Genetic Algorithm
POPULATION_SIZE = 100
NUM_GENERATIONS = 200
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 5


# Calculate distance between two cities
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# Calculate total distance for a route
def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += calculate_distance(cities[route[i]], cities[route[i + 1]])
    total_distance += calculate_distance(cities[route[-1]], cities[route[0]])
    return total_distance


# Fitness of a route
def fitness(route):
    return 1 / calculate_total_distance(route)


# Initialize the population
def initialize_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(random.sample(list(cities.keys()), len(cities)))
    return population


# Tournament selection
def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness, reverse=True)
    return tournament[0]


# Ordered crossover
def ordered_crossover(parent1, parent2):
    if np.random.rand() > CROSSOVER_RATE:
        return parent1
    child = [''] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    pos = end
    for city in itertools.chain(parent2[end:], parent2[:end]):
        if city not in child:
            child[pos % len(child)] = city
            pos += 1
    return child


# Mutation
def mutate(route):
    if np.random.rand() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


# Main Genetic Algorithm
def genetic_algorithm():
    # Initialize the population
    population = initialize_population()
    best_route = min(population, key=calculate_total_distance)
    best_distance = calculate_total_distance(best_route)

    for generation in range(NUM_GENERATIONS):
        new_population = []
        population.sort(key=fitness, reverse=True)
        new_population.extend(population[:ELITISM_COUNT])  # Elitism

        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            while parent2 == parent1:
                parent2 = tournament_selection(population)

            # Create child and mutate
            child = ordered_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        current_best_route = min(population, key=calculate_total_distance)
        current_best_distance = calculate_total_distance(current_best_route)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

    return best_route, best_distance


# Run the GA and get the best route
best_route, best_distance = genetic_algorithm()
print(best_route)
print(best_distance)

import random
import numpy as np

# 20231108
class Particle:
    def __init__(self, n):
        self.position = np.array([random.uniform(0, n - 1) for _ in range(n)])
        self.velocity = np.zeros(n)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')
        self.fitness = float('inf')

    def update_velocity(self, global_best_position, w=0.72, c1=1.49, c2=1.49):
        r1 = random.random()
        r2 = random.random()
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

def fitness(position):
    discrete_position = np.round(position).astype(int)
    n = len(discrete_position)
    non_attacking = 0
    for i in range(n):
        for j in range(i + 1, n):
            if discrete_position[i] != discrete_position[j] and abs(discrete_position[i] - discrete_position[j]) != j - i:
                non_attacking += 1
    max_non_attacking = n * (n - 1) // 2
    return max_non_attacking - non_attacking

def pso(n, max_iter=1000):
    num_particles = 30
    particles = [Particle(n) for _ in range(num_particles)]

    global_best_fitness = float('inf')
    global_best_position = None

    bounds = (0, n - 1)

    for _ in range(max_iter):
        for particle in particles:
            particle.fitness = fitness(particle.position)

            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = np.copy(particle.position)

            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = np.copy(particle.position)

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)

        if global_best_fitness == 0:
            global_best_position = np.round(global_best_position).astype(int)
            break

    if global_best_fitness != 0:
        global_best_position = np.round(global_best_position).astype(int)

    return global_best_position, global_best_fitness

solution, fitness_value = pso(16)

print('Solution:', solution)
print('Attacking pairs:', fitness_value)

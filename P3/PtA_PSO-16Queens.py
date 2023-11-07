import random
import numpy as np


class Particle:
    def __init__(self, board_size):
        self.position = np.array([random.randint(0, board_size - 1) for _ in range(board_size)])
        self.velocity = np.array([0 for _ in range(board_size)])
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=1):
        inertia = w * self.velocity
        cognitive_component = c1 * random.random() * (self.best_position - self.position)
        social_component = c2 * random.random() * (global_best_position - self.position)
        self.velocity = inertia + cognitive_component + social_component

    def update_position(self, board_size):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, board_size - 1).astype(int)


def fitness(position):
    board_size = len(position)
    attacking_pairs = 0
    for i in range(board_size):
        for j in range(i + 1, board_size):
            if position[i] == position[j] or abs(position[i] - position[j]) == j - i:
                attacking_pairs += 1
    return attacking_pairs


def pso_for_n_queens(board_size, num_particles=30, max_iter=100):
    particles = [Particle(board_size) for _ in range(num_particles)]
    global_best_score = float('inf')
    global_best_position = None

    for _ in range(max_iter):
        for particle in particles:
            score = fitness(particle.position)

            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)

            if score < global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(board_size)

        if global_best_score == 0:
            break

    return global_best_position, global_best_score


board_size = 16
solution, score = pso_for_n_queens(board_size)

if score == 0:
    print("Solution found:")
    print(solution)
else:
    print("No complete solution found, best score:", score)

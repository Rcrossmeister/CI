
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

def nearest_neighbor(dist_matrix, start=0):
    num_cities = len(dist_matrix)
    unvisited = set(range(num_cities))
    current_city = start
    path = [current_city]
    unvisited.remove(current_city)
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
        path.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    path.append(start)
    return path

def compute_length(path, dist_matrix):
    return sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))

path = nearest_neighbor(dist)
length = compute_length(path, dist)


path_cities = [cities[i] for i in path]

path_str = " -> ".join(path_cities)
print(f"Path: {path_str}")
print(f"Total length: {length}")

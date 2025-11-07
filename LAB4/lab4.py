import numpy as np

# Coordinates of cities (x, y)
cities = np.array([
    [0, 0],    # City 0
    [1, 5],    # City 1
    [5, 2],    # City 2
    [6, 6],    # City 3
    [8, 3],    # City 4
])

num_cities = len(cities)

# Calculate Euclidean distance matrix
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = euclidean_distance(cities[i], cities[j])
        else:
            distance_matrix[i][j] = np.inf  # No self loops

# ACO parameters
num_ants = 10
max_iter = 100
alpha = 1.0       # pheromone importance
beta = 5.0        # heuristic importance (distance)
evaporation_rate = 0.5
Q = 100           # pheromone deposit factor

# Initialize pheromone matrix with small positive values
pheromone = np.ones((num_cities, num_cities))

# Heuristic matrix (inverse of distance)
heuristic = 1 / distance_matrix
heuristic[heuristic == np.inf] = 0  # handle diagonal

def choose_next_city(current_city, visited):
    allowed = [city for city in range(num_cities) if city not in visited]
    probabilities = []
    for city in allowed:
        tau = pheromone[current_city, city] ** alpha
        eta = heuristic[current_city, city] ** beta
        probabilities.append(tau * eta)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    next_city = np.random.choice(allowed, p=probabilities)
    return next_city

best_tour = None
best_length = float('inf')

for iteration in range(max_iter):
    all_tours = []
    all_lengths = []

    for ant in range(num_ants):
        visited = [np.random.randint(num_cities)]  # start at random city
        while len(visited) < num_cities:
            next_city = choose_next_city(visited[-1], visited)
            visited.append(next_city)
        visited.append(visited[0])  # return to start city

        # Calculate tour length
        length = sum(distance_matrix[visited[i], visited[i+1]] for i in range(num_cities))
        
        all_tours.append(visited)
        all_lengths.append(length)

        if length < best_length:
            best_length = length
            best_tour = visited

    # Evaporate pheromone
    pheromone *= (1 - evaporation_rate)
    
    # Deposit pheromone
    for tour, length in zip(all_tours, all_lengths):
        for i in range(num_cities):
            pheromone[tour[i], tour[i+1]] += Q / length

    print(f"Iteration {iteration+1}, Best length: {best_length:.4f}")

print("\nBest tour found:", best_tour)
print("Best tour length:", best_length)

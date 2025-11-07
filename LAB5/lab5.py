import numpy as np
import random

# Coordinates of cities (x, y)
cities = np.array([
    [0, 0],
    [1, 5],
    [5, 2],
    [6, 6],
    [8, 3],
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
            distance_matrix[i][j] = np.inf

# Fitness: total length of the tour
def tour_length(tour):
    return sum(distance_matrix[tour[i], tour[(i+1) % num_cities]] for i in range(num_cities))

# Generate initial population of random tours
def initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population

# Generate a new solution by swapping two cities in the tour (simple mutation)
def generate_new_solution(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(num_cities), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# Abandon worst nests with probability pa and generate new random tours
def abandon_nests(population, fitnesses, pa):
    n = len(population)
    num_abandon = int(pa * n)
    worst_indices = np.argsort(fitnesses)[-num_abandon:]
    for idx in worst_indices:
        new_tour = list(range(num_cities))
        random.shuffle(new_tour)
        population[idx] = new_tour
    return population

# Parameters
pop_size = 15
max_iter = 200
pa = 0.25  # discovery rate (fraction of nests abandoned)

# Initialize population and fitnesses
population = initial_population(pop_size)
fitnesses = [tour_length(t) for t in population]

best_idx = np.argmin(fitnesses)
best_tour = population[best_idx]
best_length = fitnesses[best_idx]

for iteration in range(max_iter):
    new_population = population.copy()
    
    # Generate new solutions for each cuckoo (random mutation)
    for i in range(pop_size):
        new_tour = generate_new_solution(population[i])
        new_length = tour_length(new_tour)
        
        # Replace if new solution is better
        if new_length < fitnesses[i]:
            new_population[i] = new_tour
            fitnesses[i] = new_length
    
    # Abandon some nests and create new ones
    new_population = abandon_nests(new_population, fitnesses, pa)
    
    # Update population and fitnesses
    population = new_population
    fitnesses = [tour_length(t) for t in population]
    
    # Update best solution
    current_best_idx = np.argmin(fitnesses)
    if fitnesses[current_best_idx] < best_length:
        best_length = fitnesses[current_best_idx]
        best_tour = population[current_best_idx]
    
    if iteration % 20 == 0 or iteration == max_iter-1:
        print(f"Iteration {iteration+1}, Best tour length: {best_length:.4f}")

print("\nBest tour found:", best_tour)
print("Best tour length:", best_length)

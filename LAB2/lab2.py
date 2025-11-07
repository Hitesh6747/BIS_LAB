import random
import numpy as np

# Target function: f(x) = x^2 + 3x + 2
def target_function(x):
    return x**2 + 3*x + 2

# Fitness function: Calculate Mean Squared Error (MSE)
def fitness(individual, data_points):
    mse = 0
    for x, y_true in data_points:
        y_pred = evaluate_expression(individual, x)
        mse += (y_pred - y_true)**2
    return mse / len(data_points)

# Evaluate the quadratic expression ax^2 + bx + c for a given x
def evaluate_expression(individual, x):
    a, b, c = individual  # Unpack the coefficients from the individual
    return a * x**2 + b * x + c

# Generate a random individual (coefficients for the quadratic equation)
def generate_individual():
    a = random.uniform(-10, 10)  # Random coefficient for x^2
    b = random.uniform(-10, 10)  # Random coefficient for x
    c = random.uniform(-10, 10)  # Random constant
    return (a, b, c)

# Tournament selection
def tournament_selection(population, fitnesses, tournament_size=3):
    selected = random.sample(range(len(population)), tournament_size)
    best_individual = min(selected, key=lambda idx: fitnesses[idx])
    return population[best_individual]

# Uniform crossover (select each coefficient randomly from the two parents)
def crossover(parent1, parent2):
    a1, b1, c1 = parent1
    a2, b2, c2 = parent2
    child1 = (
        random.choice([a1, a2]),
        random.choice([b1, b2]),
        random.choice([c1, c2])
    )
    child2 = (
        random.choice([a1, a2]),
        random.choice([b1, b2]),
        random.choice([c1, c2])
    )
    return child1, child2

# Mutation (randomly change one of the coefficients)
def mutate(individual):
    a, b, c = individual
    mutation_point = random.randint(0, 2)
    if mutation_point == 0:
        a = random.uniform(-10, 10)
    elif mutation_point == 1:
        b = random.uniform(-10, 10)
    else:
        c = random.uniform(-10, 10)
    return (a, b, c)

# Gene Expression Algorithm (GEA) to evolve the quadratic equation
def gene_expression_algorithm(pop_size, num_generations, data_points):
    population = [generate_individual() for _ in range(pop_size)]
    
    for generation in range(num_generations):
        # Evaluate fitness for each individual in the population
        fitnesses = [fitness(individual, data_points) for individual in population]
        best_individual = population[np.argmin(fitnesses)]
        print(f"Generation {generation + 1}: Best Fitness = {min(fitnesses)}")
        
        # Check if fitness is sufficiently good, if yes, exit early
        if min(fitnesses) < 0.01:
            print("Early stopping due to satisfactory fitness.")
            break
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        
        population = new_population[:pop_size]

    # Return the best individual found
    return population[np.argmin(fitnesses)]

# Target data points based on f(x) = x^2 + 3x + 2
data_points = [(x, target_function(x)) for x in range(-5, 6)]  # Test points from -5 to 5

# Run the Gene Expression Algorithm
best_solution = gene_expression_algorithm(pop_size=50, num_generations=100, data_points=data_points)
print("Best solution (quadratic coefficients):", best_solution)

import random

# === Genetic Algorithm Parameters ===
CHROMOSOME_LENGTH = 20   # Length of the binary string
POPULATION_SIZE = 50     # Number of chromosomes in each generation
GENERATIONS = 100        # Number of generations to evolve
MUTATION_RATE = 0.01     # Probability of flipping each bit during mutation
TOURNAMENT_SIZE = 3      # Number of individuals in tournament selection


# === Step 1: Initialization ===
def generate_chromosome():
    """Generate a random binary string (chromosome)."""
    return ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))


# === Step 2: Fitness Function ===
def fitness(chromosome):
    """Fitness is simply the number of 1s in the chromosome."""
    return chromosome.count('1')


# === Step 3: Selection (Tournament Selection) ===
def select_parent(population):
    """Select the best chromosome from a random subset of the population."""
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=fitness)


# === Step 4: Crossover (Single-Point) ===
def crossover(parent1, parent2):
    """Combine two parents using single-point crossover."""
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]


# === Step 5: Mutation (Bit Flip) ===
def mutate(chromosome):
    """Flip each bit with a small probability."""
    mutated = ''
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated


# === Main Genetic Algorithm ===
def genetic_algorithm():
    # Step 1: Initialize population
    population = [generate_chromosome() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        new_population = []

        # Create new population
        for _ in range(POPULATION_SIZE):
            # Step 3: Select parents
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            # Step 4: Crossover
            child = crossover(parent1, parent2)

            # Step 5: Mutation
            child = mutate(child)

            new_population.append(child)

        population = new_population

        # Reporting
        best = max(population, key=fitness)
        print(f"Generation {generation + 1}: Best = {best} (Fitness = {fitness(best)})")

        # Step 7: Termination if optimal
        if fitness(best) == CHROMOSOME_LENGTH:
            print("\nOptimal solution found!")
            break

    print("\nFinal Best Solution:", best)


# === Run the Algorithm ===
if __name__ == "__main__":
    genetic_algorithm()

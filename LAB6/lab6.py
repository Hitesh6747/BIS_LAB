import numpy as np
import random

# Number of jobs, machines, and iterations
num_jobs = 5
num_machines = 3
num_wolves = 10
iterations = 100

# Define job and machine processing time (random for simulation purposes)
# Each job has a set of operations that need to be processed on specific machines
processing_times = np.random.randint(1, 10, size=(num_jobs, num_machines))

# Initialize the wolves randomly (each wolf represents a schedule)
def initialize_wolves():
    wolves = []
    for _ in range(num_wolves):
        # Generate a random permutation of job-machine assignments
        schedule = []
        for job in range(num_jobs):
            schedule.append(random.sample(range(num_machines), num_machines))
        wolves.append(schedule)
    return wolves

# Fitness Function (Makespan Calculation)
def calculate_makespan(schedule):
    # Initialize the start and finish times for each machine
    machine_finish_times = np.zeros(num_machines)
    job_finish_times = np.zeros(num_jobs)

    for job in range(num_jobs):
        for machine in range(num_machines):
            # Find when the operation on this machine can start
            start_time = max(machine_finish_times[schedule[job][machine]], job_finish_times[job])
            finish_time = start_time + processing_times[job][schedule[job][machine]]
            
            # Update machine and job finish times
            machine_finish_times[schedule[job][machine]] = finish_time
            job_finish_times[job] = finish_time
    
    # The makespan is the finish time of the last job
    return max(job_finish_times)

# Grey Wolf Optimization for JSSP
def gwo_jssp():
    # Initialize wolves (random schedules)
    wolves = initialize_wolves()

    # Best solution variables (alpha, beta, delta wolves)
    alpha_wolf = None
    alpha_fitness = float('inf')
    beta_wolf = None
    beta_fitness = float('inf')
    delta_wolf = None
    delta_fitness = float('inf')

    # GWO parameters
    a = 2  # Decreases linearly from 2 to 0 over iterations

    for iter in range(iterations):
        for i, wolf in enumerate(wolves):
            # Calculate fitness (makespan)
            fitness = calculate_makespan(wolf)

            # Update the alpha, beta, and delta wolves
            if fitness < alpha_fitness:
                delta_fitness = beta_fitness
                delta_wolf = beta_wolf
                beta_fitness = alpha_fitness
                beta_wolf = alpha_wolf
                alpha_fitness = fitness
                alpha_wolf = wolf
            elif fitness < beta_fitness:
                delta_fitness = beta_fitness
                delta_wolf = beta_wolf
                beta_fitness = fitness
                beta_wolf = wolf
            elif fitness < delta_fitness:
                delta_fitness = fitness
                delta_wolf = wolf

        # Update wolf positions (schedules)
        for i in range(num_wolves):
            a = 2 - iter * (2 / iterations)  # Linearly decrease a

            # Update each wolf's position (schedule)
            for job in range(num_jobs):
                for machine in range(num_machines):
                    r1 = random.random()
                    r2 = random.random()

                    A = 2 * a * r1 - a
                    C = 2 * r2

                    D_alpha = abs(C * np.array(alpha_wolf[job]) - np.array(wolves[i][job]))
                    D_beta = abs(C * np.array(beta_wolf[job]) - np.array(wolves[i][job]))
                    D_delta = abs(C * np.array(delta_wolf[job]) - np.array(wolves[i][job]))

                    # Update the wolf's position (schedule) using the GWO formula
                    wolves[i][job] = [round(x) for x in wolves[i][job]]

        print(f"Iteration {iter + 1}, Alpha Fitness (Makespan): {alpha_fitness}")
        
    return alpha_wolf, alpha_fitness

# Run the GWO algorithm on JSSP
best_schedule, best_makespan = gwo_jssp()
print("Best schedule found:", best_schedule)
print("Best makespan:", best_makespan)

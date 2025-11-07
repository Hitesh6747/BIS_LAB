
import numpy as np

# Parameters
num_swimmers = 10
target = np.array([50, 50])
max_iter = 100
w = 0.5       # inertia weight
c1 = 1.5      # cognitive constant
c2 = 1.5      # social constant

# Initialize positions and velocities randomly in 2D space
positions = np.random.rand(num_swimmers, 2) * 100
velocities = np.random.rand(num_swimmers, 2) * 2 - 1  # velocities between -1 and 1

# Initialize pbest and gbest
pbest_positions = np.copy(positions)
pbest_scores = np.linalg.norm(positions - target, axis=1)
gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index]

for iteration in range(max_iter):
    for i in range(num_swimmers):
        # Update velocity
        r1, r2 = np.random.rand(2)
        velocities[i] = (w * velocities[i] + 
                         c1 * r1 * (pbest_positions[i] - positions[i]) +
                         c2 * r2 * (gbest_position - positions[i]))
        
        # Update position
        positions[i] += velocities[i]

        # Compute fitness
        fitness = np.linalg.norm(positions[i] - target)
        
        # Update personal best
        if fitness < pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest_positions[i] = positions[i].copy()
    
    # Update global best
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index]
    
    print(f"Iteration {iteration+1}, Best distance: {pbest_scores[gbest_index]:.4f}")

print("Swimmers converged near:", gbest_position)

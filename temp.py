import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate random 2D data (replace this with your actual data)
np.random.seed(0)
data = np.random.randn(1000, 2)

# Create a grid of points to evaluate the KDE
x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Perform KDE estimation
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(data)

# Evaluate the KDE on the grid of points
log_density = kde.score_samples(xy_grid)
density = np.exp(log_density).reshape(x_grid.shape)

# Plot the KDE
plt.contourf(x_grid, y_grid, density, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D KDE Clustering')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters for the oscillating sea floor
H0 = 100  # Average depth
y0 = 30   # Amplitude of oscillation
lambda0 = 50  # Wavelength of oscillation

# Grid parameters
N1 = 100  # Number of horizontal points
N3 = 30   # Number of vertical layers

# Horizontal positions
x = np.linspace(0, 200, N1)

# Sea floor depth function
h = H0 - y0 * np.sin(2 * np.pi / lambda0 * x)

# Create temperature distribution T_ik
# Here, we simulate a simple model where temperature decreases with depth
# and has some variation in x-direction.
T = np.zeros((N1, N3))
for i in range(N1):
    for k in range(N3):
        # Simulate some temperature profile (for demonstration)
        # Temperature decreases linearly with depth and has sinusoidal variation with x
        T[i, k] = 20 - 0.1 * k * (h[i] / N3) - 5 * np.sin(2 * np.pi / 100 * x[i])

# Vertical layers, scaled by the depth function
z_layers = np.zeros((N1, N3))
for i in range(N1):
    z_layers[i] = np.linspace(0, h[i], N3)

# Plotting
plt.figure(figsize=(12, 8))

# Plot each column with its associated depth and temperature
for i in range(N1):
    plt.scatter(
        np.full(N3, x[i]), z_layers[i], c=T[i], cmap='coolwarm', s=10
    )

# Plot the sea floor
plt.plot(x, h, color='black', label='Sea Floor', linewidth=2)

# Add labels and title
plt.xlabel('Horizontal Distance (x)')
plt.ylabel('Depth (z)')
plt.title('Temperature Cross Section of the Ocean')
plt.gca().invert_yaxis()
plt.colorbar(label='Temperature (°C)')
plt.grid()
plt.show()

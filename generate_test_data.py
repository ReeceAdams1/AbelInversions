import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_test_data(filename="test_data_gaussian.csv", shape=(100, 100), center=(50, 50), sigma=20, noise_level=0.0):
    """
    Generates a 2D Gaussian distribution and saves it as a CSV file.
    This simulates a plasma column with a Gaussian density profile.
    
    The analytical Abel transform of a Gaussian f(r) = exp(-r^2/sigma^2) 
    is another Gaussian F(y) = sqrt(pi) * sigma * exp(-y^2/sigma^2).
    
    So if we generate a 2D Gaussian image, the "row" profile is the projection.
    The Abel inversion of that projection should recover the original Gaussian radial profile.
    """
    y, x = np.indices(shape)
    y0, x0 = center
    
    # Radial distance from center
    r2 = (x - x0)**2 + (y - y0)**2
    
    # 2D Gaussian function (this represents the "projection" if we consider the camera sees the integrated line of sight)
    # Wait, for Abel inversion testing:
    # We usually start with a known radial distribution f(r).
    # Then we calculate the forward Abel transform to get the projection P(x).
    # The app takes P(x) (or a 2D image of P(x)) and inverts it to get f(r).
    
    # Let's generate a 2D image where each row is the same projection P(x).
    # Let's use a Gaussian radial profile: f(r) = exp(-r^2 / w^2)
    # The projection P(x) is also a Gaussian: P(x) = sqrt(pi) * w * exp(-x^2 / w^2)
    
    x_axis = np.arange(shape[1])
    rel_x = x_axis - x0
    
    # Projection P(x)
    # Amplitude A=1 for the source f(r)
    # P(x) amplitude = sqrt(pi) * sigma
    projection = np.sqrt(np.pi) * sigma * np.exp(-rel_x**2 / sigma**2)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, projection.shape)
        projection += noise
        
    # Create 2D image by repeating the row
    image_data = np.tile(projection, (shape[0], 1))
    
    # Save to CSV
    np.savetxt(filename, image_data, delimiter=',')
    print(f"Generated {filename} with shape {shape}, center at x={x0}, sigma={sigma}")
    
    return x_axis, projection

def generate_test_data_step(filename="test_data_step.csv", shape=(100, 100), center=(50, 50), radius=20, noise_level=0.0):
    """
    Generates a "Top Hat" or Step function profile.
    Source f(r) = 1 if r < R, else 0.
    Projection P(x) = 2 * sqrt(R^2 - x^2)
    """
    x_axis = np.arange(shape[1])
    x0 = center[1]
    rel_x = x_axis - x0
    
    # Projection P(x)
    projection = np.zeros_like(rel_x, dtype=float)
    mask = np.abs(rel_x) < radius
    projection[mask] = 2 * np.sqrt(radius**2 - rel_x[mask]**2)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, projection.shape)
        projection += noise
        
    # Create 2D image
    image_data = np.tile(projection, (shape[0], 1))
    
    np.savetxt(filename, image_data, delimiter=',')
    print(f"Generated {filename} with shape {shape}, center at x={x0}, radius={radius}")

if __name__ == "__main__":
    # 1. Gaussian Profile (Smooth)
    # Expected Inversion: Gaussian
    generate_test_data("test_gaussian.csv", shape=(200, 500), center=(100, 250), sigma=50, noise_level=0.0)
    
    # 2. Step Profile (Sharp Edge)
    # Expected Inversion: Top Hat (Rectangular function)
    generate_test_data_step("test_step.csv", shape=(200, 500), center=(100, 250), radius=50, noise_level=0.0)
    
    # 3. Noisy Gaussian
    generate_test_data("test_gaussian_noisy.csv", shape=(200, 500), center=(100, 250), sigma=50, noise_level=5.0)

    print("\nTest files created. Load these into the Abel Inversion App.")

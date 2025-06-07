#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Load data
def load_path(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)

# Load paths
orig_x, orig_y = load_path('original_path.csv')
replan_x, replan_y = load_path('replanned_path.csv')

# Create plot
plt.figure(figsize=(12, 10))

# Plot map background
map_img = plt.imread('map.png')
plt.imshow(map_img, cmap='gray', extent=[0, map_img.shape[1], 0, map_img.shape[0]], alpha=0.7)

# Plot paths
plt.plot(orig_x, orig_y, 'b-', linewidth=2, label='Original Path')
plt.plot(replan_x, replan_y, 'g--', linewidth=2, label='Replanned Path')
plt.plot(orig_x[0], orig_y[0], 'go', markersize=10, label='Start')
plt.plot(orig_x[-1], orig_y[-1], 'ro', markersize=10, label='Goal')

# Add legend and labels
plt.legend()
plt.title('Path Planning Visualization')
plt.xlabel('X (grid cells)')
plt.ylabel('Y (grid cells)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# Save and show
plt.savefig('path_comparison.png', dpi=300)
plt.show()

"""
Data Investigation Script
Investigate the structure of the loaded data to understand the label format
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Investigating dataset structure...")
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"X dtype: {X.dtype}")
print(f"Y dtype: {Y.dtype}")

print(f"\nX stats:")
print(f"  Min: {X.min()}")
print(f"  Max: {X.max()}")
print(f"  Mean: {X.mean():.4f}")

print(f"\nY analysis:")
print(f"First 10 Y values:")
print(Y[:10])

print(f"\nY unique values per sample (should be 1 for one-hot):")
print(f"First 5 samples sum: {Y[:5].sum(axis=1)}")

# Check if Y is already one-hot encoded
if Y.ndim == 2 and Y.shape[1] == 10:
    print(f"\nY appears to be one-hot encoded with {Y.shape[1]} classes")
    # Convert one-hot to class labels
    y_labels = np.argmax(Y, axis=1)
    print(f"Converted class labels shape: {y_labels.shape}")
    print(f"Unique classes: {np.unique(y_labels)}")
    
    # Count samples per class
    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"\nClass distribution:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} samples")
else:
    print(f"\nY is not one-hot encoded")

# Display a few sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(X[i], cmap='gray')
    if Y.ndim == 2:
        digit = np.argmax(Y[i])
    else:
        digit = Y[i]
    axes[row, col].set_title(f'Sample {i}: Digit {digit}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('data_investigation.png', dpi=150, bbox_inches='tight')
plt.show()
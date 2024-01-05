import numpy as np

def generate_offsets(axis, n = 50, sigma = 0.1):
    offsets = np.random.normal(loc = 0, scale = sigma, size = (n, 6))
    offsets[:, 0] = 0
    np.savetxt("offsets_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

if __name__ == "__main__":
    offsets_y = generate_offsets(axis = "y", n = 50, sigma = 0.1)
    offsets_z = generate_offsets(axis = "z", n = 50, sigma = 0.1)
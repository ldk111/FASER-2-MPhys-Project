import numpy as np

def generate_offsets(axis, n = 50, sigma = 0.1):
    offsets = np.random.normal(loc = 0, scale = sigma, size = (n, 6))
    offsets[:, 0] = 0
    np.savetxt("offsets_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

def generate_test_offsets(axis):
    offsets = np.zeros((12, 6))
    if axis == "y":
        for i in range(0, 6):
            offsets[i,i] = 0.2
    else:
        for i in range(6, 12):
            offsets[i,i-6] = 0.2
    np.savetxt("offsets_test_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

if __name__ == "__main__":
    offsets_y = generate_offsets(axis = "y", n = 50, sigma = 0.2)
    offsets_z = generate_offsets(axis = "z", n = 50, sigma = 0.2)
    offsets_x = generate_offsets(axis = "x", n = 50, sigma = 0.2)

    offsets_test_y = generate_test_offsets("y")
    offsets_test_z = generate_test_offsets("z")
    offsets_test_x = generate_test_offsets("x")
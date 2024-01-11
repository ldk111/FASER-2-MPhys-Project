import numpy as np

def generate_offsets(axis, n = 50, sigma = 0.1, outdir = ""):
    offsets = np.random.normal(loc = 0, scale = sigma, size = (n, 6))
    offsets[:, 0] = 0
    offsets[:, 3] = 0
    offsets[:, 4] = 0
    offsets[:, 5] = 0
    np.savetxt(outdir + "offsets_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

def generate_blank_offsets(axis, n = 50, outdir = ""):
    offsets = np.zeros((n, 6))
    np.savetxt(outdir + "offsets_" + axis + ".csv", offsets, delimiter = ",")

def generate_test_offsets(axis, outdir = ""):
    offsets = np.zeros((12, 6))
    if axis == "y":
        for i in range(0, 6):
            offsets[i,i] = 0.05
    elif axis == "z":
        for i in range(6, 12):
            offsets[i,i-6] = 0.05
    np.savetxt(outdir + "offsets_test_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

if __name__ == "__main__":
    offsets_y = generate_offsets(axis = "y", n = 50, sigma = 0.0625, outdir = "")
    offsets_z = generate_offsets(axis = "z", n = 50, sigma = 0.0625, outdir = "")
    offsets_x = generate_blank_offsets(axis = "x", n = 50, outdir = "")

    #offsets_test_y = generate_test_offsets("y", outdir = "variable_test/")
    #offsets_test_z = generate_test_offsets("z", outdir = "variable_test/")
    #offsets_test_x = generate_test_offsets("x", outdir = "variable_test/")
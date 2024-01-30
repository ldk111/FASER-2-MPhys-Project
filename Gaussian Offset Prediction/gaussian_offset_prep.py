import numpy as np

def generate_offsets(axis, n = 50, sigma = 0.0625, outdir = ""):
    offsets = np.random.normal(loc = 0, scale = sigma, size = (n, 6))
    offsets[:, 0] = 0
    #offsets[:, 3] = 0
    #offsets[:, 4] = 0
    #offsets[:, 5] = 0
    np.savetxt(outdir + "offsets_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

def generate_rotations(axis, n = 50, sigma = np.pi/32, outdir = ""):
    offsets = np.random.normal(loc = 0, scale = sigma, size = (n, 6))
    offsets[:, 0] = 0
    #offsets[:, 3] = 0
    #offsets[:, 4] = 0
    #offsets[:, 5] = 0
    np.savetxt(outdir + "rotations_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

def generate_blank_offsets(axis, n = 50, outdir = ""):
    offsets = np.zeros((n, 6))
    np.savetxt(outdir + "offsets_" + axis + ".csv", offsets, delimiter = ",")

def generate_blank_rotations(axis, n = 50, outdir = ""):
    offsets = np.zeros((n, 6))
    np.savetxt(outdir + "rotations_" + axis + ".csv", offsets, delimiter = ",")

def generate_test_offsets(axis, outdir = ""):
    offsets = np.zeros((6, 6))
    if axis == "x":
        for i in range(0, 6):
            offsets[i,i] = 0.05
    elif axis == "z":
        for i in range(6, 12):
            offsets[i,i-6] = 0.05
    np.savetxt(outdir + "offsets_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

def generate_test_rotations(axis, outdir = ""):
    offsets = np.zeros((18, 6))
    if axis == "x":
        for i in range(0, 6):
            offsets[i,i] = np.pi/8
    elif axis == "y":
        for i in range(6, 12):
            offsets[i,i-6] = np.pi/8
    elif axis == "z":
        for i in range(12, 18):
            offsets[i,i-12] = np.pi/8
    np.savetxt(outdir + "rotations_" + axis + ".csv", offsets, delimiter = ",")
    return offsets

if __name__ == "__main__":
    #offsets_y = generate_blank_offsets(axis = "y", n = 6, outdir = "")
    #generate_blank_offsets(axis = "x", n = 50, outdir = "")
    #offsets_x = generate_test_offsets(axis = "x", outdir = "")

    generate_rotations(axis = "x")
    generate_rotations(axis = "y")
    generate_rotations(axis = "z")

    generate_offsets(axis = "x")
    generate_offsets(axis = "y")
    generate_offsets(axis = "z")

    #generate_test_rotations("x")
    #generate_test_rotations("y")
    #generate_test_rotations("z")

    #generate_blank_rotations("x", n=50)
    #generate_blank_rotations("y", n=50)
    #generate_blank_rotations("z", n=50)


    #offsets_test_y = generate_test_offsets("y", outdir = "variable_test/")
    #offsets_test_z = generate_test_offsets("z", outdir = "variable_test/")
    #offsets_test_x = generate_test_offsets("x", outdir = "variable_test/")
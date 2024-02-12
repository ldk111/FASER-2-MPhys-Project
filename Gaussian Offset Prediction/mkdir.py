import os

def Generate_Directories(n_iterations, n_offsets, n_inputs, path):

    for i in range(0, n_iterations):

        try:
            os.mkdir(path + str(i))
        except FileExistsError:
            print("File Exists")

        for j in range(0, n_offsets):

            try:
                os.mkdir(path + str(i) + "/" + str(j))
            except FileExistsError:
                print("File Exists")

            for k in range(0, n_inputs):

                try:
                    os.mkdir(path + str(i) + "/" + str(j) + "/" + str(k))
                except FileExistsError:
                    print("File Exists")

    return print("Directories successfully created for " + str(n_offsets) + " offsets and " + str(n_inputs) + " inputs.")
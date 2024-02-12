import sys
sys.path.insert(0, "/home/chri6112/pythonPackages")

import recreate_acts_input as recreate

n_samples = 50
n_inputs = 6
input_dir = ""

for i in range(0, n_samples):
    for j in range(0, n_inputs):
        outdir = str(i) + "/repropagation.root"
        input_dir = str(i) + "/"
        recreate.Recreate_Input(input_dir, j, outdir)
    
    
import sys

sys.path.insert(0, "/home/chri6112/pythonPackages")

import acts
import evaluation as eval
import itertools
import concurrent.futures
import numpy as np

from kalman_alignedTel import runTruthTrackingKalman, runACTS, runACTSreprop
from mkdir import Generate_Directories

def main():

    u = acts.UnitConstants

    rand_ = np.random.randint(0, 50000)

    digiConfigFile= "/data/atlassmallfiles/users/chri6112/Acts/acts/Examples/Algorithms/Digitization/share/default-smearing-config-telescope.json"
    inputParticlePath = "/data/atlassmallfiles/users/chri6112/Muon Flux Data/Run Data/muon_filter_200k_"

    field = acts.RestrictedBField(acts.Vector3(0* u.T, 0, 1.0 * u.T))

    detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
        bounds=[500, 1500], positions=[10000, 10500, 11000, 19500, 20000, 20500], binValue=0,thickness=4,
    )

    outputPath = "output/100224_10_200k@200k_yz_13_6/"
    n_offsets = 5
    n_inputs = 1
    n_iterations = 2
    reprop = False

    offsetIndex = np.repeat(np.arange(0, n_offsets), n_inputs)
    inputIndex = np.tile(np.arange(0,n_inputs), n_offsets)

    #Extra iteration added here so new predicted output can be stored but not analysed
    Generate_Directories(n_iterations+1, n_offsets, n_inputs, outputPath)

    for i in range(0, n_iterations):

        #Load in new offsets
        offsets_x = np.loadtxt(outputPath + str(i) + "/offsets_x.csv", delimiter = ",")
        offsets_y = np.loadtxt(outputPath + str(i) + "/offsets_y.csv", delimiter = ",")
        offsets_z = np.loadtxt(outputPath + str(i) + "/offsets_z.csv", delimiter = ",")

        rotations_x = np.loadtxt(outputPath + str(i) + "/rotations_x.csv", delimiter=",")
        rotations_y = np.loadtxt(outputPath + str(i) + "/rotations_y.csv", delimiter=",")
        rotations_z = np.loadtxt(outputPath + str(i) + "/rotations_z.csv", delimiter=",")

        #Run ACTS for this configuration of offsets
        input_args = list(zip(offsetIndex, inputIndex, 
                                np.repeat(offsets_x, n_inputs, axis = 0), np.repeat(offsets_y, n_inputs, axis = 0), np.repeat(-offsets_z, n_inputs, axis = 0), 
                                np.repeat(rotations_x, n_inputs, axis = 0), np.repeat(rotations_y, n_inputs, axis = 0), np.repeat(-rotations_z, n_inputs, axis = 0), 
                                itertools.repeat(field), itertools.repeat(digiConfigFile), itertools.repeat(trackingGeometry), itertools.repeat(inputParticlePath), itertools.repeat(rand_), itertools.repeat(outputPath + str(i) + "/")))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = 100) as executor:
            executor.map(runACTS, input_args)

        #Select subset of offsets as returned by analysis
        offsets_x = offsets_x[:,1:7]
        offsets_y = offsets_y[:,1:7]
        offsets_z = offsets_z[:,1:7]

        rotations_x = rotations_x[:,1:7]
        rotations_y = rotations_y[:,1:7]
        rotations_z = rotations_z[:,1:7]

        #Analyse results of ACTS run from original folder
        truth_parameters = np.array(np.array([rotations_x[0:n_offsets], rotations_y[0:n_offsets], rotations_z[0:n_offsets], offsets_x[0:n_offsets], offsets_y[0:n_offsets], offsets_z[0:n_offsets]])).transpose((1,2,0))

        parameters, sum_of_residuals_squared, sum_of_no_squared, df, df_offsets = eval.Analyse_Multiple_Runs(outputPath + str(i) + "/", n_offsets, n_inputs,
                                                                            offsets_x = offsets_x, offsets_y = offsets_y, offsets_z = offsets_z, 
                                                                            rotations_x = rotations_x, rotations_y = rotations_y, rotations_z = rotations_z,
                                                                            initial_guess = np.array([0.0,0.0,0.0,0.0,0.1,-0.1]),
                                                                            bounds = [(0, 0),(0, 0),(0, 0),(0, 0),(-1, 1),(-1, 1)],
                                                                            reprop=False)
        
        #Generate updated parameters
        new_rotations_x = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,0]-parameters[:,:,0], axis = 1)
        new_rotations_y = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,1]-parameters[:,:,1], axis = 1)
        new_rotations_z = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,2]-parameters[:,:,2], axis = 1)

        new_offsets_x = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,3]-parameters[:,:,3], axis = 1)
        new_offsets_y = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,4]-parameters[:,:,4], axis = 1)
        new_offsets_z = np.append(np.zeros((n_offsets,1)),truth_parameters[:,:,5]-parameters[:,:,5], axis = 1)
    
        pred_rotations_x = np.append(np.zeros((n_offsets,1)),parameters[:,:,0], axis = 1)
        pred_rotations_y = np.append(np.zeros((n_offsets,1)),parameters[:,:,1], axis = 1)
        pred_rotations_z = np.append(np.zeros((n_offsets,1)),parameters[:,:,2], axis = 1)

        pred_offsets_x = np.append(np.zeros((n_offsets,1)),parameters[:,:,3], axis = 1)
        pred_offsets_y = np.append(np.zeros((n_offsets,1)),parameters[:,:,4], axis = 1)
        pred_offsets_z = np.append(np.zeros((n_offsets,1)),parameters[:,:,5], axis = 1)
        
        #Save updated parameters in new folder for next run
        np.savetxt(outputPath + str(i+1) + "/offsets_x.csv", new_offsets_x, delimiter = ",")
        np.savetxt(outputPath + str(i+1) + "/offsets_y.csv", new_offsets_y, delimiter = ",")
        np.savetxt(outputPath + str(i+1) + "/offsets_z.csv", new_offsets_z, delimiter = ",")

        np.savetxt(outputPath + str(i+1) + "/rotations_x.csv", new_rotations_x, delimiter = ",")
        np.savetxt(outputPath + str(i+1) + "/rotations_y.csv", new_rotations_y, delimiter = ",")
        np.savetxt(outputPath + str(i+1) + "/rotations_z.csv", new_rotations_z, delimiter = ",")

        #Save predictions in output of current run folder
        np.savetxt(outputPath + str(i) + "/pred_offsets_x.csv", pred_offsets_x, delimiter = ",")
        np.savetxt(outputPath + str(i) + "/pred_offsets_y.csv", pred_offsets_y, delimiter = ",")
        np.savetxt(outputPath + str(i) + "/pred_offsets_z.csv", pred_offsets_z, delimiter = ",")

        np.savetxt(outputPath + str(i) + "/pred_rotations_x.csv", pred_rotations_x, delimiter = ",")
        np.savetxt(outputPath + str(i) + "/pred_rotations_y.csv", pred_rotations_y, delimiter = ",")
        np.savetxt(outputPath + str(i) + "/pred_rotations_z.csv", pred_rotations_z, delimiter = ",")
               
if __name__ == "__main__":

    main()

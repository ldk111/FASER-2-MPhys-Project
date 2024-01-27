import numpy as np
import pandas as pd

import simulation
import reconstruction

def Analyse_Run(input_dir, i, offsets_x, offsets_y, offsets_z, rotations_x, rotations_y, rotations_z, initial_guess = np.random.uniform(-0.01, 0.01, 6), bounds = [(-np.pi/8, np.pi/8),(-np.pi/8, np.pi/8),(-np.pi/8, np.pi/8),(-1, 1),(-1, 1),(-1, 1)]):
    """
    Analyses a single run from a given directory, returns the fitted parameters for each plane along with a sum of squares of the residuals between the measured and transformed coordinates.

    Parameters:
    - input_dir: The path to the main folder of the ACTS run, with folders representing each run inside.
    - i: The index of the run being analysed, the specific subfolder within input_dir.
    - offsets_x: 2D array with the offsets of the tracking planes along the x axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - offsets_y: 2D array with the offsets of the tracking planes along the y axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - offsets_z: 2D array with the offsets of the tracking planes along the z axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - rotations_x: 2D array with the rotations of the tracking planes along the x axis in radians, currently rotates the hit positions not the detector.
    - rotations_y: 2D array with the rotations of the tracking planes along the y axis in radians, currently rotates the hit positions not the detector.
    - rotations_z: 2D array with the rotations of the tracking planes along the z axis in radians, currently rotates the hit positions not the detector.
    - initial_guess: Initial guess for the transformation parameters.
    - bounds: Lower and upper bounds for each of the parameters [tx, ty, tz, dx, dy, dz], can set to (0,0) if you want to eliminate a transformation.

    Returns:
    - total_parameters: 2D array with the fitted parameters for each of the planes.
    - total_sum_of_residuals_squared: Array with the sum of the residuals squared for each plane.
    """

    #Initialise empty arrays to store outputs for each tracking plane in
    total_parameters = []
    total_sum_of_residuals_squared = []

    #Creates a dataframe from the ROOT files from the run
    df = simulation.Generate_DataFrame_From_ROOT(input_dir, i)

    #Creates the offset dataframe
    df_offsets = simulation.Generate_Predicted_Offset_DataFrame(df)

    #For each of the planes 2 through 6 (1 is fixed to have 0 offset), fit the parameters and retrieve the sum of residuals squared 
    for plane in range(2, 7):

        parameters, sum_of_residuals_squared = reconstruction.Fit_Offsets(df_offsets, plane, initial_guess, bounds)

        #Add the results of this plane's results to the total results
        total_parameters.append([parameters])
        total_sum_of_residuals_squared.append([sum_of_residuals_squared])

    return total_parameters, total_sum_of_residuals_squared, df, df_offsets


def Analyse_Multiple_Runs(input_dir, n_samples, offsets_x, offsets_y, offsets_z, rotations_x, rotations_y, rotations_z):
    """
    Analyses a single run from a given directory, returns the fitted parameters for each plane along with a sum of squares of the residuals between the measured and transformed coordinates.

    Parameters:
    - input_dir: The path to the main folder of the ACTS run, with folders representing each run inside.
    - n_samples: The number of runs within input_dir.
    - offsets_x: 2D array with the offsets of the tracking planes along the x axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - offsets_y: 2D array with the offsets of the tracking planes along the y axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - offsets_z: 2D array with the offsets of the tracking planes along the z axis in mm, such that the global hit = predicted hit + offset (CHECK SIGN HERE).
    - rotations_x: 2D array with the rotations of the tracking planes along the x axis in radians, currently rotates the hit positions not the detector.
    - rotations_y: 2D array with the rotations of the tracking planes along the y axis in radians, currently rotates the hit positions not the detector.
    - rotations_z: 2D array with the rotations of the tracking planes along the z axis in radians, currently rotates the hit positions not the detector.

    Returns:
    - pred_parameters: 2D array with the fitted parameters for each run and plane within the run.
    - total_sum_of_residuals_squared: Array with the sum of the residuals squared for each run and plane within the run.
    - df_offsets: A dataframe containing the offsets dataframe for each of the runs appended on to each other to get the total offsets dataframe for all the runs, useful if runs have same offsets and different input.
    """

    #Initialise empty arrays to store outputs for each tracking plane in
    pred_parameters = np.array([])
    total_sum_of_residuals_squared = np.array([])

    #Not the best way to do this, desired output is a cumulative offset dataframe so multiple runs with different inpus and same offsets (ACTS doesn't like more than 200k events in a run) can be analysed together
    first = True

    #For each run within the input_dir
    for i in range(0, n_samples):

        print("\nANALYSING DATAFRAME: " + str(i) + "\n")

        #Retrieve the results from analysing a single run, we do not specify bounds or initial guesses here currently, instead we use the default ones from Analyse_Run
        pred_parameters_i, total_sum_of_residuals_squared_i, _df, _df_offsets = Analyse_Run(input_dir, i, offsets_y[i], offsets_z[i])

        #If this is the first run we store the output offsets dataframe as what is returned
        if first == True:
            first = False
            df_offsets = _df_offsets
        #If this is not the first run, we append the new offsets dataframe to the current offsets dataframe
        else:
            df_offsets = pd.concat([df_offsets, _df_offsets], ignore_index=True)

        #Add the fitted parameters (sum of residuals squared) onto the total list of fitted parameters (sum of residuals squared)
        pred_parameters = np.append(pred_parameters, pred_parameters_i)
        total_sum_of_residuals_squared = np.append(total_sum_of_residuals_squared, total_sum_of_residuals_squared_i)

    #This splits the data into n_samples different rows so each row is a single run of ACTS containing the data for 5 tracking planes (not 6 as 1st plane is not fitted since it is fixed as a reference plane)
    pred_parameters = np.array(np.hsplit(pred_parameters, n_samples))
    total_sum_of_residuals_squared = np.array(np.hsplit(total_sum_of_residuals_squared, n_samples))

    #Save results into the same directory as the main run
    np.savetxt(input_dir + "pred_parameters.csv", pred_parameters, delimiter = ",")
    np.savetxt(input_dir + "total_sum_of_residuals_squared.csv", total_sum_of_residuals_squared, delimiter = ",")

    return pred_parameters, total_sum_of_residuals_squared, df_offsets
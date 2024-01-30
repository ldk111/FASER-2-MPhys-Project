import pandas as pd
import numpy as np
import uproot
import awkward as ak

def Shift_NaNs(input_array):
    """
    Shifts the NaNs to the beginning of the array rather than end.
    ACTS output has all real results at the beginning rather than end of array.
    We want all hits to be in the correct column, so shift the measured hits to the end where they should be.

    Parameters:
    - input_array: An input array with 6 columns and many rows, with NaNs needing to be shifted.

    Returns:
    - output_array: The input array with NaNs shifted to the start.
    """
    
    nan_mask = np.isnan(input_array)

    #Essentially this swaps the NaNs and numbers by first selecting only the NaNs, then not the NaNs and concatenating them together
    output_array = np.array([np.concatenate((row[nan_mask_row], row[~nan_mask_row])) for row, nan_mask_row in zip(input_array, nan_mask)])

    return output_array

def Process_Coords(input_ak_array):
    """
    Processes a coordinate input from ACTS trackstates.root file into a dataframe compatible format.
    Shifts all the NaNs and makes sure that each row has 6 columns with NaNs in place of missing data.
    Final output has structure of each row is a particle, and then column 0 is plane 6, column 1 is plane 5...

    Parameters:
    - input_ak_array: An input awkward array read directly from ACTS trackstates.root using uproot.

    Returns:
    - output_array: The result of applying Shift_NaNs to the inputted awkward array after it has been padded with NaNs so each row has 6 entries (for each plane).
    """
    return Shift_NaNs(ak.to_numpy(ak.pad_none(input_ak_array, target = 6, clip=True)).filled(np.nan))
    
def Generate_DataFrame_From_ROOT(input_dir, i):
    """
    Creates a pandas dataframe from the output of a run of kalman_alignedTel.py

    Parameters:
    - input_dir: The path to the main folder of the ACTS run, with folders representing each run inside.
    - i: The index of the run being analysed, the specific subfolder within input_dir.

    Returns:
    - df: The pandas dataframe with columns required for analysis, for details see below.
    """

    #Save results to the input directory with dataframe labelled by index
    output_path = input_dir + "df" + str(i) + ".csv"

    #Set path for ACTS results
    trackstates_path = input_dir + str(i) + "/trackstates_fitter.root"
    tracksummary_path = input_dir + str(i) + "/tracksummary_fitter.root"

    #Open trackstates first and extract any data we want measurements for at each plane
    file = uproot.open(trackstates_path)
    tree_input = file["trackstates"]

    #Each variable we want to anaylse is extracted as an awkward array and then processed using Process_Coords into a numpy array
    #To add more variables look within the ROOT file and add the column as demonstrated below
    X_TRUTH = Process_Coords(tree_input["t_x"].array())
    GLOBAL_X_HIT = Process_Coords(tree_input["g_x_hit"].array())
    FIT_X_HIT = Process_Coords(tree_input["g_x_smt"].array())

    Y_TRUTH = Process_Coords(tree_input["t_y"].array())
    GLOBAL_Y_HIT = Process_Coords(tree_input["g_y_hit"].array())
    FIT_Y_HIT = Process_Coords(tree_input["g_y_smt"].array())
    LOCAL_Y_HIT = Process_Coords(tree_input["l_y_hit"].array())

    Z_TRUTH = Process_Coords(tree_input["t_z"].array())
    GLOBAL_Z_HIT = Process_Coords(tree_input["g_z_hit"].array())
    FIT_Z_HIT = Process_Coords(tree_input["g_z_smt"].array())
    LOCAL_Z_HIT = Process_Coords(-tree_input["l_x_hit"].array())

    FIT_PX = Process_Coords(tree_input["px_smt"].array())
    FIT_PY = Process_Coords(tree_input["py_smt"].array())
    FIT_PZ = Process_Coords(tree_input["pz_smt"].array())

    file.close()

    #Repeat process with tracksummary, this time each array contains only one entry as value is for particle not per station
    file = uproot.open(tracksummary_path)
    tree_input = file["tracksummary"]

    #Each variable is extracted as an array, flattened to 1D and then converted to a numpy array
    #Again to add more variables look within the ROOT file and add below as demonstrated
    QOP_FIT = ak.to_numpy(ak.flatten(tree_input["eQOP_fit"].array()))
    PHI_FIT = ak.to_numpy(ak.flatten(tree_input["ePHI_fit"].array()))
    THETA_FIT = ak.to_numpy(ak.flatten(tree_input["eTHETA_fit"].array()))

    P_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_p"].array()))
    Q_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_charge"].array()))
    PX_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_px"].array()))
    PY_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_py"].array()))
    PZ_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_pz"].array()))
    PHI_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_phi"].array()))
    THETA_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_theta"].array()))

    CHI2SUM = ak.to_numpy(ak.flatten(tree_input["chi2Sum"].array()))
    NDF = ak.to_numpy(ak.flatten(tree_input["NDF"].array()))

    file.close()

    #This might need to be slightly edited due to using Q_TRUTH, should look for a Q_FIT value
    P_FIT = Q_TRUTH/QOP_FIT

    #Add any more derived quantities wanted here
    PZ_FIT = P_FIT*np.cos(THETA_FIT)

    #Specifies the columns we want in our dataframe, generally try and match the names to variables extracted above
    df_columns = [
                "QOP_FIT", "PHI_FIT", "THETA_FIT", "P_FIT", "PZ_FIT", 
                "P_TRUTH", "Q_TRUTH", "PX_TRUTH", "PY_TRUTH", "PZ_TRUTH", "PHI_TRUTH", "THETA_TRUTH", 
                "CHI2SUM", "NDF", 
                "FIT_PX_6", "FIT_PX_5", "FIT_PX_4", "FIT_PX_3", "FIT_PX_2", "FIT_PX_1",
                "FIT_PY_6", "FIT_PY_5", "FIT_PY_4", "FIT_PY_3", "FIT_PY_2", "FIT_PY_1",
                "FIT_PZ_6", "FIT_PZ_5", "FIT_PZ_4", "FIT_PZ_3", "FIT_PZ_2", "FIT_PZ_1", 
                "X_TRUTH_6", "X_TRUTH_5", "X_TRUTH_4", "X_TRUTH_3", "X_TRUTH_2", "X_TRUTH_1", 
                "GLOBAL_X_HIT_6", "GLOBAL_X_HIT_5", "GLOBAL_X_HIT_4", "GLOBAL_X_HIT_3", "GLOBAL_X_HIT_2", "GLOBAL_X_HIT_1",
                "FIT_X_HIT_6", "FIT_X_HIT_5", "FIT_X_HIT_4", "FIT_X_HIT_3", "FIT_X_HIT_2", "FIT_X_HIT_1", 
                "Y_TRUTH_6", "Y_TRUTH_5", "Y_TRUTH_4", "Y_TRUTH_3", "Y_TRUTH_2", "Y_TRUTH_1", 
                "GLOBAL_Y_HIT_6", "GLOBAL_Y_HIT_5", "GLOBAL_Y_HIT_4", "GLOBAL_Y_HIT_3", "GLOBAL_Y_HIT_2", "GLOBAL_Y_HIT_1", 
                "LOCAL_Y_HIT_6", "LOCAL_Y_HIT_5", "LOCAL_Y_HIT_4", "LOCAL_Y_HIT_3", "LOCAL_Y_HIT_2", "LOCAL_Y_HIT_1", 
                "FIT_Y_HIT_6", "FIT_Y_HIT_5", "FIT_Y_HIT_4", "FIT_Y_HIT_3", "FIT_Y_HIT_2", "FIT_Y_HIT_1",
                "Z_TRUTH_6", "Z_TRUTH_5", "Z_TRUTH_4", "Z_TRUTH_3", "Z_TRUTH_2", "Z_TRUTH_1", 
                "GLOBAL_Z_HIT_6", "GLOBAL_Z_HIT_5", "GLOBAL_Z_HIT_4", "GLOBAL_Z_HIT_3", "GLOBAL_Z_HIT_2", "GLOBAL_Z_HIT_1",
                "LOCAL_Z_HIT_6", "LOCAL_Z_HIT_5", "LOCAL_Z_HIT_4", "LOCAL_Z_HIT_3", "LOCAL_Z_HIT_2", "LOCAL_Z_HIT_1",
                "FIT_Z_HIT_6", "FIT_Z_HIT_5", "FIT_Z_HIT_4", "FIT_Z_HIT_3", "FIT_Z_HIT_2", "FIT_Z_HIT_1"
    ]
    
    #Adds the data to the dataframe, note that the output of Process_Coords returns the coordinate data in the reverse order to what you might expect
    #1st column is 6th tracking plane, 2nd column 5th tracking plane etc
    df_data = [
                QOP_FIT, PHI_FIT, THETA_FIT, P_FIT, PZ_FIT,
                P_TRUTH, Q_TRUTH, PX_TRUTH, PY_TRUTH, PZ_TRUTH, PHI_TRUTH, THETA_TRUTH, 
                CHI2SUM, NDF, 
                FIT_PX[:,0], FIT_PX[:,1], FIT_PX[:,2], FIT_PX[:,3], FIT_PX[:,4], FIT_PX[:,5], 
                FIT_PY[:,0], FIT_PY[:,1], FIT_PY[:,2], FIT_PY[:,3], FIT_PY[:,4], FIT_PY[:,5], 
                FIT_PZ[:,0], FIT_PZ[:,1], FIT_PZ[:,2], FIT_PZ[:,3], FIT_PZ[:,4], FIT_PZ[:,5], 
                X_TRUTH[:,0], X_TRUTH[:,1], X_TRUTH[:,2], X_TRUTH[:,3], X_TRUTH[:,4], X_TRUTH[:,5], 
                GLOBAL_X_HIT[:,0], GLOBAL_X_HIT[:,1], GLOBAL_X_HIT[:,2], GLOBAL_X_HIT[:,3], GLOBAL_X_HIT[:,4], GLOBAL_X_HIT[:,5], 
                FIT_X_HIT[:,0], FIT_X_HIT[:,1], FIT_X_HIT[:,2], FIT_X_HIT[:,3], FIT_X_HIT[:,4], FIT_X_HIT[:,5],  
                Y_TRUTH[:,0], Y_TRUTH[:,1], Y_TRUTH[:,2], Y_TRUTH[:,3], Y_TRUTH[:,4], Y_TRUTH[:,5], 
                GLOBAL_Y_HIT[:,0], GLOBAL_Y_HIT[:,1], GLOBAL_Y_HIT[:,2], GLOBAL_Y_HIT[:,3], GLOBAL_Y_HIT[:,4], GLOBAL_Y_HIT[:,5],
                LOCAL_Y_HIT[:,0], LOCAL_Y_HIT[:,1], LOCAL_Y_HIT[:,2], LOCAL_Y_HIT[:,3], LOCAL_Y_HIT[:,4], LOCAL_Y_HIT[:,5],
                FIT_Y_HIT[:,0], FIT_Y_HIT[:,1], FIT_Y_HIT[:,2], FIT_Y_HIT[:,3], FIT_Y_HIT[:,4], FIT_Y_HIT[:,5],  
                Z_TRUTH[:,0], Z_TRUTH[:,1], Z_TRUTH[:,2], Z_TRUTH[:,3], Z_TRUTH[:,4], Z_TRUTH[:,5], 
                GLOBAL_Z_HIT[:,0], GLOBAL_Z_HIT[:,1], GLOBAL_Z_HIT[:,2], GLOBAL_Z_HIT[:,3], GLOBAL_Z_HIT[:,4], GLOBAL_Z_HIT[:,5],
                LOCAL_Z_HIT[:,0], LOCAL_Z_HIT[:,1], LOCAL_Z_HIT[:,2], LOCAL_Z_HIT[:,3], LOCAL_Z_HIT[:,4], LOCAL_Z_HIT[:,5],
                FIT_Z_HIT[:,0], FIT_Z_HIT[:,1], FIT_Z_HIT[:,2], FIT_Z_HIT[:,3], FIT_Z_HIT[:,4], FIT_Z_HIT[:,5],  
    ]

    #Creates a dataframe from the above dictionary and array
    df = pd.DataFrame(data=np.column_stack(df_data), columns=df_columns)

    #This removes any rows in place that did not reach the end of the detector by specifying that the hit position in the final plane is not NaN
    df.dropna(subset=["GLOBAL_Z_HIT_6"], inplace=True, ignore_index=True)

    #Saves dataframe as CSV
    df.to_csv(output_path)

    return df

def Propagate_B_Field(px, py, pz, e, B, dx):
    """
    Propagates a particle through the magnetic field specified within ACTS.
    Returns the final momentum after the magnetic field along with the changes in y, z, and x.
    Note our ACTS field is uniform and along the z direction so that is the geometry used here.

    Parameters:
    - px: The momentum of the particle along the x direction in GeV/c.
    - py: The momentum of the particle along the y direction in GeV/c.
    - pz: The momentum of the particle along the z direction in GeV/c.
    - e: The charge of the particle in units of -e, i.e 1 for a muon, -1 for antimuon.
    - B: The magnetic flux density in Tesla.
    - dx: The length of the magnetic field in mm.

    Returns:
    - px_prime: The momentum of the particle along the x direction after the magnetic field in GeV/c.
    - py_prime: The momentum of the particle along the x direction after the magnetic field in GeV/c.
    - pz_prime: The momentum of the particle along the x direction after the magnetic field in GeV/c.
    - dx: The change in x of the particle in mm (same as inputted).
    - dy: The change in y of the particle in mm.
    - dz: The change in z of the particle in mm.

    """

    #Magnetic field along z does not affect momentum along z
    pz_prime = pz
    
    #Specifying required constants, c in m/s, e in Coulombs
    c = 299792458
    e = 1.60217663*10**(-19)*e

    #Converting dx to metres
    dx = dx/1000

    #Calculates the time the particle is present in the magnetic field (see Notion or OneNote for derivation)
    wt = -np.arccos((dx*B*c/10**9 + py)/(np.sqrt(px**2 + py**2))) + np.arctan2(px, py)

    #Calculates the final momentum values in GeV/c based on time spent in magnetic field
    px_prime = (px * np.cos(wt) - py * np.sin(wt))
    py_prime = (px * np.sin(wt) + py * np.cos(wt))

    #Reconverts dx to mm
    dx = dx*1000

    #Calculates change in position in mm based on time spent in magnetic field
    dy = (- px/(B*c/10**9) * np.cos(wt) + py/(B*c/10**9) * np.sin(wt) + px/(B*c/10**9))*1000
    dz = (wt/(B*c/10**9) * pz) * 1000

    return px_prime, py_prime, pz_prime, dx, dy, dz

def Generate_Predicted_Offset_DataFrame(df):
    """
    Creates a pandas dataframe with the predicted hit positions after simulating the muons propagating through the detector.
    Uses the expected detector geometry and Propagate_B_Field function to propagate what is measured in the first plane through as if everything else was aligned.

    Parameters:
    - df: The pandas dataframe output of Generate_DataFrame_From_ROOT containing all the raw parameters from the ACTS run.

    Returns:
    - output_df: The pandas dataframe with columns containing the measured and predicted hit locations.
    """

    #Initialise empty dataframe with the columns we want to have in our output dataframe, add columns here if more variables are needed
    output_dict = {
                    "HIT_Y_1": [], "HIT_Y_2": [], "HIT_Y_3": [], "HIT_Y_4": [], "HIT_Y_5": [], "HIT_Y_6": [], 
                    "PRED_Y_1": [], "PRED_Y_2": [], "PRED_Y_3": [], "PRED_Y_4": [], "PRED_Y_5": [], "PRED_Y_6": [], 
 
                    "HIT_Z_1": [], "HIT_Z_2": [], "HIT_Z_3": [], "HIT_Z_4": [], "HIT_Z_5": [], "HIT_Z_6": [], 
                    "PRED_Z_1": [], "PRED_Z_2": [], "PRED_Z_3": [], "PRED_Z_4": [], "PRED_Z_5": [], "PRED_Z_6": [],

                    #"FIT_PX_1": df["FIT_PX_1"].values, "FIT_PX_2": df["FIT_PX_2"].values, "FIT_PX_3": df["FIT_PX_3"].values, "FIT_PX_4": df["FIT_PX_4"].values, "FIT_PX_5": df["FIT_PX_5"].values, "FIT_PX_6": df["FIT_PX_6"].values,
                    #"FIT_PX_1": df["FIT_PY_1"].values, "FIT_PY_2": df["FIT_PY_2"].values, "FIT_PY_3": df["FIT_PY_3"].values, "FIT_PY_4": df["FIT_PY_4"].values, "FIT_PY_5": df["FIT_PY_5"].values, "FIT_PY_6": df["FIT_PY_6"].values,
                    #"FIT_PX_1": df["FIT_PZ_1"].values, "FIT_PZ_2": df["FIT_PZ_2"].values, "FIT_PZ_3": df["FIT_PZ_3"].values, "FIT_PZ_4": df["FIT_PZ_4"].values, "FIT_PZ_5": df["FIT_PZ_5"].values, "FIT_PZ_6": df["FIT_PZ_6"].values,
                    
                    "FIT_PX_1": df["FIT_PY_1"], "FIT_PX_2": df["FIT_PY_1"], "FIT_PX_3": df["FIT_PY_1"], "FIT_PX_4": [], "FIT_PX_5": [], "FIT_PX_6": [],
                    "FIT_PX_1": df["FIT_PZ_1"], "FIT_PY_2": df["FIT_PZ_1"], "FIT_PY_3": df["FIT_PZ_1"], "FIT_PY_4": [], "FIT_PY_5": [], "FIT_PY_6": [],
                    "FIT_PX_1": df["FIT_PX_1"], "FIT_PZ_2": df["FIT_PX_1"], "FIT_PZ_3": df["FIT_PX_1"], "FIT_PZ_4": [], "FIT_PZ_5": [], "FIT_PZ_6": [],
    }

    #Initial tracking plane is assumed to be aligned correctly
    output_dict["PRED_Y_1"] = df["GLOBAL_Y_HIT_1"]
    output_dict["PRED_Z_1"] = df["GLOBAL_Z_HIT_1"]

    #For each tracking plane, we extract the measured hit positions
    for i in range(1, 7):

        output_dict["HIT_Y_" + str(i)] = df["GLOBAL_Y_HIT_" + str(i)]
        output_dict["HIT_Z_" + str(i)] = df["GLOBAL_Z_HIT_" + str(i)]

    #Defining input variables for simulation, using momentum as fitted for the first plane
    PY = df["FIT_PY_1"]
    PZ = df["FIT_PZ_1"]
    PX = df["FIT_PX_1"]

    #These are the changes in y and z respectively between the first 3 tracking planes, predicted by the fitted momentum given the distance between planes
    delta_y = PY/PX * 500
    delta_z = PZ/PX * 500

    #Propagating the first 3 tracking planes
    for i in range(1, 3):

        #Particles should be moving in a straight line and just receive multiples of delta_coordinate added on to their original position
        PRED_Y = i * delta_y + df["GLOBAL_Y_HIT_1"]
        PRED_Z = i * delta_z + df["GLOBAL_Z_HIT_1"]

        #Store these outputs in the relevant dataframe column
        output_dict["PRED_Y_" + str(i+1)] = PRED_Y
        output_dict["PRED_Z_" + str(i+1)] = PRED_Z 

    #Propagate particles through the B field and retrieve their positions and momenta after the B field
    PX_PRIME, PY_PRIME, PZ_PRIME, DX, DY, DZ = Propagate_B_Field(PX, PY, PZ, 1, 1, 4000)

    #Calculate new changes in y and z between the final 3 tracking planes
    delta_y_prime = PY_PRIME/PX_PRIME * 500
    delta_z_prime = PZ_PRIME/PX_PRIME * 500

    #Propagate the final 3 tracking planes
    for i in range(3, 6):

        #Particles travels a total of 1500mm before hitting the B field (3 * delta_y), goes through B field (DY), travels 4000mm to first tracking plane with new momentum (4000*PY_PRIME/PX_PRIME), and 500 mm between subsequent tracking planes (delta_y_prime)
        PRED_Y = 3 * delta_y + df["GLOBAL_Y_HIT_1"] + DY + 4000 * PY_PRIME/PX_PRIME + (i-3) * delta_y_prime 
        PRED_Z = 3 * delta_z + df["GLOBAL_Z_HIT_1"] + DZ + 4000 * PZ_PRIME/PX_PRIME + (i-3) * delta_z_prime 

        output_dict["PRED_Y_" + str(i+1)] = PRED_Y
        output_dict["PRED_Z_" + str(i+1)] = PRED_Z 

        output_dict["FIT_PX_" + str(i+1)] = PX_PRIME
        output_dict["FIT_PY_" + str(i+1)] = PY_PRIME
        output_dict["FIT_PZ_" + str(i+1)] = PZ_PRIME
        
    #Creates output dataframe from dictionary    
    output_df = pd.DataFrame(output_dict)

    return output_df
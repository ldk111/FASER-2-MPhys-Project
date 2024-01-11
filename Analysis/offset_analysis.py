import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import uproot
import awkward as ak

def Shift_NaNs(input_array):
    
    nan_mask = np.isnan(input_array)
    output_array = np.array([np.concatenate((row[nan_mask_row], row[~nan_mask_row])) for row, nan_mask_row in zip(input_array, nan_mask)])

    return output_array

def Process_Coords(input_ak_array):
    return Shift_NaNs(ak.to_numpy(ak.pad_none(input_ak_array, target = 6, clip=True)).filled(np.nan))
    
def Generate_DataFrame_From_ROOT(input_dir, i):

    output_path = input_dir + "df" + str(i) + ".csv"
    trackstates_path = input_dir + str(i) + "/trackstates_fitter.root"
    tracksummary_path = input_dir + str(i) + "/tracksummary_fitter.root"

    file = uproot.open(trackstates_path)
    tree_input = file["trackstates"]

    X_TRUTH = Process_Coords(tree_input["t_x"].array())
    GLOBAL_X_HIT = Process_Coords(tree_input["g_x_hit"].array())
    FIT_X_HIT = Process_Coords(tree_input["g_x_smt"].array())

    Y_TRUTH = Process_Coords(tree_input["t_y"].array())
    GLOBAL_Y_HIT = Process_Coords(tree_input["g_y_hit"].array())
    FIT_Y_HIT = Process_Coords(tree_input["g_y_smt"].array())

    Z_TRUTH = Process_Coords(tree_input["t_z"].array())
    GLOBAL_Z_HIT = Process_Coords(tree_input["g_z_hit"].array())
    FIT_Z_HIT = Process_Coords(tree_input["g_z_smt"].array())

    FIT_PX = Process_Coords(tree_input["px_smt"].array())
    FIT_PY = Process_Coords(tree_input["py_smt"].array())
    FIT_PZ = Process_Coords(tree_input["pz_smt"].array())

    file.close()

    file = uproot.open(tracksummary_path)
    tree_input = file["tracksummary"]

    QOP_FIT = ak.to_numpy(ak.flatten(tree_input["eQOP_fit"].array()))
    PHI_FIT = ak.to_numpy(ak.flatten(tree_input["ePHI_fit"].array()))
    THETA_FIT = ak.to_numpy(ak.flatten(tree_input["eTHETA_fit"].array()))

    P_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_p"].array()))
    Q_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_charge"].array()))
    PZ_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_pz"].array()))
    PHI_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_phi"].array()))
    THETA_TRUTH = ak.to_numpy(ak.flatten(tree_input["t_theta"].array()))

    CHI2SUM = ak.to_numpy(ak.flatten(tree_input["chi2Sum"].array()))
    NDF = ak.to_numpy(ak.flatten(tree_input["NDF"].array()))

    file.close()

    P_FIT = Q_TRUTH/QOP_FIT
    PZ_FIT = P_FIT*np.cos(THETA_FIT)

    df_columns = [
                "QOP_FIT", "PHI_FIT", "THETA_FIT", "P_FIT", "PZ_FIT", 
                "P_TRUTH", "Q_TRUTH", "PZ_TRUTH", "PHI_TRUTH", "THETA_TRUTH", 
                "CHI2SUM", "NDF", 
                "FIT_PX_6", "FIT_PX_5", "FIT_PX_4", "FIT_PX_3", "FIT_PX_2", "FIT_PX_1",
                "FIT_PY_6", "FIT_PY_5", "FIT_PY_4", "FIT_PY_3", "FIT_PY_2", "FIT_PY_1",
                "FIT_PZ_6", "FIT_PZ_5", "FIT_PZ_4", "FIT_PZ_3", "FIT_PZ_2", "FIT_PZ_1", 
                "X_TRUTH_6", "X_TRUTH_5", "X_TRUTH_4", "X_TRUTH_3", "X_TRUTH_2", "X_TRUTH_1", 
                "GLOBAL_X_HIT_6", "GLOBAL_X_HIT_5", "GLOBAL_X_HIT_4", "GLOBAL_X_HIT_3", "GLOBAL_X_HIT_2", "GLOBAL_X_HIT_1",
                "FIT_X_HIT_6", "FIT_X_HIT_5", "FIT_X_HIT_4", "FIT_X_HIT_3", "FIT_X_HIT_2", "FIT_X_HIT_1", 
                "Y_TRUTH_6", "Y_TRUTH_5", "Y_TRUTH_4", "Y_TRUTH_3", "Y_TRUTH_2", "Y_TRUTH_1", 
                "GLOBAL_Y_HIT_6", "GLOBAL_Y_HIT_5", "GLOBAL_Y_HIT_4", "GLOBAL_Y_HIT_3", "GLOBAL_Y_HIT_2", "GLOBAL_Y_HIT_1", 
                "FIT_Y_HIT_6", "FIT_Y_HIT_5", "FIT_Y_HIT_4", "FIT_Y_HIT_3", "FIT_Y_HIT_2", "FIT_Y_HIT_1",
                "Z_TRUTH_6", "Z_TRUTH_5", "Z_TRUTH_4", "Z_TRUTH_3", "Z_TRUTH_2", "Z_TRUTH_1", 
                "GLOBAL_Z_HIT_6", "GLOBAL_Z_HIT_5", "GLOBAL_Z_HIT_4", "GLOBAL_Z_HIT_3", "GLOBAL_Z_HIT_2", "GLOBAL_Z_HIT_1",
                "FIT_Z_HIT_6", "FIT_Z_HIT_5", "FIT_Z_HIT_4", "FIT_Z_HIT_3", "FIT_Z_HIT_2", "FIT_Z_HIT_1"
    ]

    df_data = [
                QOP_FIT, PHI_FIT, THETA_FIT, P_FIT, PZ_FIT,
                P_TRUTH, Q_TRUTH, PZ_TRUTH, PHI_TRUTH, THETA_TRUTH, 
                CHI2SUM, NDF, 
                FIT_PX[:,0], FIT_PX[:,1], FIT_PX[:,2], FIT_PX[:,3], FIT_PX[:,4], FIT_PX[:,5], 
                FIT_PY[:,0], FIT_PY[:,1], FIT_PY[:,2], FIT_PY[:,3], FIT_PY[:,4], FIT_PY[:,5], 
                FIT_PZ[:,0], FIT_PZ[:,1], FIT_PZ[:,2], FIT_PZ[:,3], FIT_PZ[:,4], FIT_PZ[:,5], 
                X_TRUTH[:,0], X_TRUTH[:,1], X_TRUTH[:,2], X_TRUTH[:,3], X_TRUTH[:,4], X_TRUTH[:,5], 
                GLOBAL_X_HIT[:,0], GLOBAL_X_HIT[:,1], GLOBAL_X_HIT[:,2], GLOBAL_X_HIT[:,3], GLOBAL_X_HIT[:,4], GLOBAL_X_HIT[:,5], 
                FIT_X_HIT[:,0], FIT_X_HIT[:,1], FIT_X_HIT[:,2], FIT_X_HIT[:,3], FIT_X_HIT[:,4], FIT_X_HIT[:,5],  
                Y_TRUTH[:,0], Y_TRUTH[:,1], Y_TRUTH[:,2], Y_TRUTH[:,3], Y_TRUTH[:,4], Y_TRUTH[:,5], 
                GLOBAL_Y_HIT[:,0], GLOBAL_Y_HIT[:,1], GLOBAL_Y_HIT[:,2], GLOBAL_Y_HIT[:,3], GLOBAL_Y_HIT[:,4], GLOBAL_Y_HIT[:,5],
                FIT_Y_HIT[:,0], FIT_Y_HIT[:,1], FIT_Y_HIT[:,2], FIT_Y_HIT[:,3], FIT_Y_HIT[:,4], FIT_Y_HIT[:,5],  
                Z_TRUTH[:,0], Z_TRUTH[:,1], Z_TRUTH[:,2], Z_TRUTH[:,3], Z_TRUTH[:,4], Z_TRUTH[:,5], 
                GLOBAL_Z_HIT[:,0], GLOBAL_Z_HIT[:,1], GLOBAL_Z_HIT[:,2], GLOBAL_Z_HIT[:,3], GLOBAL_Z_HIT[:,4], GLOBAL_Z_HIT[:,5],
                FIT_Z_HIT[:,0], FIT_Z_HIT[:,1], FIT_Z_HIT[:,2], FIT_Z_HIT[:,3], FIT_Z_HIT[:,4], FIT_Z_HIT[:,5],  
    ]

    df = pd.DataFrame(data=np.column_stack(df_data), columns=df_columns)
    df.to_csv(output_path)

    return df

import numpy as np
from scipy.stats import norm

def Fit_Gaussian(x):

    x_no_nan = x[~np.isnan(x)]
    mu, std = norm.fit(x_no_nan)

    return x_no_nan, mu, std

def Residual_Plot(x, label = "", save = False, bins=100):
    """
    Residual_Plot takes a 1D input of data and plots it as a frequency density histogram, overlaying a fitted normal distribution.

    Inputs
    x: 1D input data, Pandas series or Numpy array
    label: adds labels to the x axis and file name if save is set to true, string
    save: if True will save the plot as label_residual_plot.png, boolean
    bins: number of bins for the histogram, integer

    Returns
    mu: the mean of the fitted normal distribution, float
    std: the standard deviation of the fitted normal distribution, float
    fig: the matplotlib figure containing the final graph, matplotlib figure
    """

    x, mu, std = Fit_Gaussian(x)

    norm_x = np.arange(start = np.min(x), stop = np.max(x), step = 0.0001)
    norm_y = norm.pdf(norm_x, mu, std)

    fig = plt.figure(figsize = (4, 4), dpi = 200)
    plt.hist(x, bins = bins, density = True)
    plt.plot(norm_x, norm_y)
    
    if label != "":
        plt.xlabel("Residual in " + label)

    plt.ylabel("Frequency Density")
    plt.text(x = -0.5, y = -1.2, s = "Mean : " + str(mu) + " mm" + "\nSigma : " + str(std) + " mm", size = 10)
    plt.show()

    if save == True:
        plt.savfig(label + "_residual_plot.png")

    return mu, std, fig

def Generate_Predicted_Offset_DataFrame(df):

    output_dict = {
                    "T_OFFSET_Y_1": [], "T_OFFSET_Y_2": [], "T_OFFSET_Y_3": [], "T_OFFSET_Y_4": [], "T_OFFSET_Y_5": [], "T_OFFSET_Y_6": [], 
                    "PRED_OFFSET_Y_1": [], "PRED_OFFSET_Y_2": [], "PRED_OFFSET_Y_3": [], "PRED_OFFSET_Y_4": [], "PRED_OFFSET_Y_5": [], "PRED_OFFSET_Y_6": [], 
                    "HIT_Y_1": [], "HIT_Y_2": [], "HIT_Y_3": [], "HIT_Y_4": [], "HIT_Y_5": [], "HIT_Y_6": [], 
                    "PRED_Y_1": [], "PRED_Y_2": [], "PRED_Y_3": [], "PRED_Y_4": [], "PRED_Y_5": [], "PRED_Y_6": [], 
                    "FIT_Y_1": [], "FIT_Y_2": [], "FIT_Y_3": [], "FIT_Y_4": [], "FIT_Y_5": [], "FIT_Y_6": [], 
 
                    "T_OFFSET_Z_1": [], "T_OFFSET_Z_2": [], "T_OFFSET_Z_3": [], "T_OFFSET_Z_4": [], "T_OFFSET_Z_5": [], "T_OFFSET_Z_6": [], 
                    "PRED_OFFSET_Z_1": [], "PRED_OFFSET_Z_2": [], "PRED_OFFSET_Z_3": [], "PRED_OFFSET_Z_4": [], "PRED_OFFSET_Z_5": [], "PRED_OFFSET_Z_6": [], 
                    "HIT_Z_1": [], "HIT_Z_2": [], "HIT_Z_3": [], "HIT_Z_4": [], "HIT_Z_5": [], "HIT_Z_6": [], 
                    "PRED_Z_1": [], "PRED_Z_2": [], "PRED_Z_3": [], "PRED_Z_4": [], "PRED_Z_5": [], "PRED_Z_6": [],
                    "FIT_Z_1": [], "FIT_Z_2": [], "FIT_Z_3": [], "FIT_Z_4": [], "FIT_Z_5": [], "FIT_Z_6": [], 
    }

    for i in range(1, 7):

        output_dict["T_OFFSET_Y_" + str(i)] = df["Y_TRUTH_" + str(i)] - df["GLOBAL_Y_HIT_" + str(i)]
        output_dict["HIT_Y_" + str(i)] = df["GLOBAL_Y_HIT_" + str(i)]
        output_dict["FIT_Y_" + str(i)] = df["FIT_Y_HIT_" + str(i)]

        output_dict["T_OFFSET_Z_" + str(i)] = df["Z_TRUTH_" + str(i)] - df["GLOBAL_Z_HIT_" + str(i)]
        output_dict["HIT_Z_" + str(i)] = df["GLOBAL_Z_HIT_" + str(i)]
        output_dict["FIT_Z_" + str(i)] = df["FIT_Z_HIT_" + str(i)]

        output_dict["PRED_Y_" + str(i)] = np.zeros(len(df))
        output_dict["PRED_OFFSET_Y_" + str(i)] = np.zeros(len(df))
        
        output_dict["PRED_Z_" + str(i)] = np.zeros(len(df))
        output_dict["PRED_OFFSET_Z_" + str(i)] = np.zeros(len(df))

    delta_y = df["FIT_PY_1"]/df["FIT_PX_1"]*(df["GLOBAL_X_HIT_2"].mean() - df["GLOBAL_X_HIT_1"].mean())
    delta_z = df["FIT_PZ_1"]/df["FIT_PX_1"]*(df["GLOBAL_X_HIT_2"].mean() - df["GLOBAL_X_HIT_1"].mean())

    for i in range(1, 3):

        PRED_Y = i * delta_y + df["GLOBAL_Y_HIT_1"]
        PRED_Z = i * delta_z + df["GLOBAL_Z_HIT_1"]

        output_dict["PRED_Y_" + str(i+1)] = PRED_Y
        output_dict["PRED_OFFSET_Y_" + str(i+1)] = PRED_Y - df["GLOBAL_Y_HIT_" + str(i+1)]

        output_dict["PRED_Z_" + str(i+1)] = PRED_Z 
        output_dict["PRED_OFFSET_Z_" + str(i+1)] = PRED_Z - df["GLOBAL_Z_HIT_" + str(i+1)]
        
    output_df = pd.DataFrame(output_dict)

    return output_df

def Summarise_DataFrame(df, offsets_y, offsets_z, plots = False):

    SUM_OF_SQUARES = 0
    SUM_OF_TRUE_SQUARES = 0
    REL_ERR_ARRAY = [] 
    PRED_OFFSET_Y_ARRAY = []
    PRED_OFFSET_Z_ARRAY = []

    for i in range(0, 6):

        PRED_OFFSET_Y = np.mean(df["PRED_OFFSET_Y_" + str(i+1)][np.abs(df["PRED_OFFSET_Y_" + str(i+1)]) < 1])
        PRED_OFFSET_Z = np.mean(df["PRED_OFFSET_Z_" + str(i+1)][np.abs(df["PRED_OFFSET_Z_" + str(i+1)]) < 1])

        TRUE_OFFSET_Y = offsets_y[i]
        TRUE_OFFSET_Z = offsets_z[i]

        RESIDUAL_Y = np.abs(PRED_OFFSET_Y) - np.abs(TRUE_OFFSET_Y)
        RESIDUAL_Z = np.abs(PRED_OFFSET_Z) - np.abs(TRUE_OFFSET_Z)

        if TRUE_OFFSET_Y != 0:
            REL_ERR_Y = (PRED_OFFSET_Y - TRUE_OFFSET_Y)/TRUE_OFFSET_Y * 100
        else:
            REL_ERR_Y = np.nan

        if TRUE_OFFSET_Z != 0:
            REL_ERR_Z = (PRED_OFFSET_Z - TRUE_OFFSET_Z)/TRUE_OFFSET_Z * 100
        else:
            REL_ERR_Z = np.nan

        print("OFFSET RECONSTRUCTION SUMMARY FOR TRACKING PLANE " + str(i+1))
        print("\nY AXIS")
        print("PREDICTED OFFSET: " + str(PRED_OFFSET_Y))
        print("TRUE OFFSET: " + str(TRUE_OFFSET_Y))
        print("RESIDUAL: " + str(RESIDUAL_Y))
        print("PERCENTAGE DIFFERENCE: " + str(REL_ERR_Y))
        print("\nZ AXIS")
        print("PREDICTED OFFSET: " + str(PRED_OFFSET_Z))
        print("TRUE OFFSET: " + str(TRUE_OFFSET_Z))
        print("RESIDUAL: " + str(RESIDUAL_Z))
        print("PERCENTAGE DIFFERENCE: " + str(REL_ERR_Z))
        print("\n")

        SUM_OF_SQUARES += RESIDUAL_Y**2 + RESIDUAL_Z**2
        SUM_OF_TRUE_SQUARES += TRUE_OFFSET_Y**2 + TRUE_OFFSET_Z**2

        REL_ERR_ARRAY.append(REL_ERR_Y)
        REL_ERR_ARRAY.append(REL_ERR_Z)
        PRED_OFFSET_Y_ARRAY.append(PRED_OFFSET_Y)
        PRED_OFFSET_Z_ARRAY.append(PRED_OFFSET_Z)

        if plots == True:
            Residual_Plot(df["PRED_OFFSET_Y_" + str(i+1)], label = "Predicted Y Offset: Plane " + str(i+1))
            Residual_Plot(df["PRED_OFFSET_Z_" + str(i+1)], label = "Predicted Z Offset: Plane " + str(i+1))

    print("OVERALL RECONSTRUCTION RESULTS")
    print("SUM OF SQUARES: " + str(SUM_OF_SQUARES))
    print("SUM OF TRUE SQUARES: " + str(SUM_OF_TRUE_SQUARES))
    print("MEAN PERCENTAGE DIFFERENCE: " + str(np.mean(REL_ERR_ARRAY)))
    print("MEDIAN PERCENTAGE DIFFERENCE: " + str(np.median(REL_ERR_ARRAY)))

    return np.array(PRED_OFFSET_Y_ARRAY), np.array(PRED_OFFSET_Z_ARRAY)

def Analyse_Run(input_dir, i, offsets_y, offsets_z, plots = False):

    df = Generate_DataFrame_From_ROOT(input_dir, i)

    df = df[df["CHI2SUM"] < 50]

    df_offsets = Generate_Predicted_Offset_DataFrame(df)

    pred_offsets_y, pred_offsets_z = Summarise_DataFrame(df_offsets, offsets_y, offsets_z, plots)

    return pred_offsets_y, pred_offsets_z

def Analyse_Multiple_Runs(input_dir, n_samples, offsets_y, offsets_z, plots = False):
    
    pred_offsets_y = np.array([])
    pred_offsets_z = np.array([])

    for i in range(0, n_samples):

        print("\nANALYSING DATAFRAME: " + str(i) + "\n")

        pred_offsets_y_i, pred_offsets_z_i = Analyse_Run(input_dir, i, offsets_y[i], offsets_z[i], plots)

        pred_offsets_y = np.append(pred_offsets_y, pred_offsets_y_i)
        pred_offsets_z = np.append(pred_offsets_z, pred_offsets_z_i)

    np.savetxt(input_dir + "pred_offsets_y_.csv", pred_offsets_y, delimiter = ",")
    np.savetxt(input_dir + "pred_offsets_z_.csv", pred_offsets_z, delimiter = ",")

    return pred_offsets_y, pred_offsets_z
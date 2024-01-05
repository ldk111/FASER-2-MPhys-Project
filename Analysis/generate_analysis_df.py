import sys
sys.path.insert(0, "/home/chri6112/pythonPackages")

import uproot
import pandas as pd
import numpy as np
import awkward as ak

output_path = "Output/200k_0.2_misal_test/200k_0.2_misal_df.csv"
trackstates_path = "Output/200k_0.2_misal_test/trackstates_fitter.root"
tracksummary_path = "Output/200k_0.2_misal_test/tracksummary_fitter.root"

file = uproot.open(trackstates_path)
tree_input = file["trackstates"]

X_TRUTH = ak.to_numpy(ak.pad_none(tree_input["t_x"].array(), target = 6, clip=True)).filled(np.nan)
GLOBAL_X_HIT = ak.to_numpy(ak.pad_none(tree_input["g_x_hit"].array(), target = 6, clip=True)).filled(np.nan)
LOCAL_X_HIT = ak.to_numpy(ak.pad_none(tree_input["l_x_hit"].array(), target = 6, clip=True)).filled(np.nan)

Y_TRUTH = ak.to_numpy(ak.pad_none(tree_input["t_y"].array(), target = 6, clip=True)).filled(np.nan)
GLOBAL_Y_HIT = ak.to_numpy(ak.pad_none(tree_input["g_y_hit"].array(), target = 6, clip=True)).filled(np.nan)
LOCAL_Y_HIT = ak.to_numpy(ak.pad_none(tree_input["l_y_hit"].array(), target = 6, clip=True)).filled(np.nan)

Z_TRUTH = ak.to_numpy(ak.pad_none(tree_input["t_z"].array(), target = 6, clip=True)).filled(np.nan)
GLOBAL_Z_HIT = ak.to_numpy(ak.pad_none(tree_input["g_z_hit"].array(), target = 6, clip=True)).filled(np.nan)

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
              "X_TRUTH_1", "X_TRUTH_2", "X_TRUTH_3", "X_TRUTH_4", "X_TRUTH_5", "X_TRUTH_6", 
              "GLOBAL_X_HIT_1", "GLOBAL_X_HIT_2", "GLOBAL_X_HIT_3", "GLOBAL_X_HIT_4", "GLOBAL_X_HIT_5", "GLOBAL_X_HIT_6", 
              "LOCAL_X_HIT_1", "LOCAL_X_HIT_2", "LOCAL_X_HIT_3", "LOCAL_X_HIT_4", "LOCAL_X_HIT_5", "LOCAL_X_HIT_6", 
              "Y_TRUTH_1", "Y_TRUTH_2", "Y_TRUTH_3", "Y_TRUTH_4", "Y_TRUTH_5", "Y_TRUTH_6", 
              "GLOBAL_Y_HIT_1", "GLOBAL_Y_HIT_2", "GLOBAL_Y_HIT_3", "GLOBAL_Y_HIT_4", "GLOBAL_Y_HIT_5", "GLOBAL_Y_HIT_6", 
              "LOCAL_Y_HIT_1", "LOCAL_Y_HIT_2", "LOCAL_Y_HIT_3", "LOCAL_Y_HIT_4", "LOCAL_Y_HIT_5", "LOCAL_Y_HIT_6", 
              "Z_TRUTH_1", "Z_TRUTH_2", "Z_TRUTH_3", "Z_TRUTH_4", "Z_TRUTH_5", "Z_TRUTH_6", 
              "GLOBAL_Z_HIT_1", "GLOBAL_Z_HIT_2", "GLOBAL_Z_HIT_3", "GLOBAL_Z_HIT_4", "GLOBAL_Z_HIT_5", "GLOBAL_Z_HIT_6"
]

df_data = [
            QOP_FIT, PHI_FIT, THETA_FIT, P_FIT, PZ_FIT,
            P_TRUTH, Q_TRUTH, PZ_TRUTH, PHI_TRUTH, THETA_TRUTH, 
            CHI2SUM, NDF, 
            X_TRUTH[:,0], X_TRUTH[:,1], X_TRUTH[:,2], X_TRUTH[:,3], X_TRUTH[:,4], X_TRUTH[:,5], 
            GLOBAL_X_HIT[:,0], GLOBAL_X_HIT[:,1], GLOBAL_X_HIT[:,2], GLOBAL_X_HIT[:,3], GLOBAL_X_HIT[:,4], GLOBAL_X_HIT[:,5], 
            LOCAL_X_HIT[:,0], LOCAL_X_HIT[:,1], LOCAL_X_HIT[:,2], LOCAL_X_HIT[:,3], LOCAL_X_HIT[:,4], LOCAL_X_HIT[:,5], 
            Y_TRUTH[:,0], Y_TRUTH[:,1], Y_TRUTH[:,2], Y_TRUTH[:,3], Y_TRUTH[:,4], Y_TRUTH[:,5], 
            GLOBAL_Y_HIT[:,0], GLOBAL_Y_HIT[:,1], GLOBAL_Y_HIT[:,2], GLOBAL_Y_HIT[:,3], GLOBAL_Y_HIT[:,4], GLOBAL_Y_HIT[:,5], 
            LOCAL_Y_HIT[:,0], LOCAL_Y_HIT[:,1], LOCAL_Y_HIT[:,2], LOCAL_Y_HIT[:,3], LOCAL_Y_HIT[:,4], LOCAL_Y_HIT[:,5], 
            Z_TRUTH[:,0], Z_TRUTH[:,1], Z_TRUTH[:,2], Z_TRUTH[:,3], Z_TRUTH[:,4], Z_TRUTH[:,5], 
            GLOBAL_Z_HIT[:,0], GLOBAL_Z_HIT[:,1], GLOBAL_Z_HIT[:,2], GLOBAL_Z_HIT[:,3], GLOBAL_Z_HIT[:,4], GLOBAL_Z_HIT[:,5]
]

df = pd.DataFrame(data=np.column_stack(df_data), columns=df_columns)
df.to_csv(output_path)
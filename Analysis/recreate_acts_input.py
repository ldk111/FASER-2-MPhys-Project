import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT as r
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

    FIT_ETA = Process_Coords(tree_input["eta_smt"].array())
    FIT_PT =Process_Coords(tree_input["pT_smt"].array())

    file.close()

    file = uproot.open(tracksummary_path)
    tree_input = file["tracksummary"]

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

    P_FIT = Q_TRUTH/QOP_FIT
    PZ_FIT = P_FIT*np.cos(THETA_FIT)

    df_columns = [
                "QOP_FIT", "PHI_FIT", "THETA_FIT", "ETA_FIT", "PT_FIT", "P_FIT", "PZ_FIT", 
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
                "FIT_Y_HIT_6", "FIT_Y_HIT_5", "FIT_Y_HIT_4", "FIT_Y_HIT_3", "FIT_Y_HIT_2", "FIT_Y_HIT_1",
                "Z_TRUTH_6", "Z_TRUTH_5", "Z_TRUTH_4", "Z_TRUTH_3", "Z_TRUTH_2", "Z_TRUTH_1", 
                "GLOBAL_Z_HIT_6", "GLOBAL_Z_HIT_5", "GLOBAL_Z_HIT_4", "GLOBAL_Z_HIT_3", "GLOBAL_Z_HIT_2", "GLOBAL_Z_HIT_1",
                "FIT_Z_HIT_6", "FIT_Z_HIT_5", "FIT_Z_HIT_4", "FIT_Z_HIT_3", "FIT_Z_HIT_2", "FIT_Z_HIT_1"
    ]

    df_data = [
                QOP_FIT, PHI_FIT, THETA_FIT, FIT_ETA[:,5], FIT_PT[:,5], P_FIT, PZ_FIT,
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
                FIT_Y_HIT[:,0], FIT_Y_HIT[:,1], FIT_Y_HIT[:,2], FIT_Y_HIT[:,3], FIT_Y_HIT[:,4], FIT_Y_HIT[:,5],  
                Z_TRUTH[:,0], Z_TRUTH[:,1], Z_TRUTH[:,2], Z_TRUTH[:,3], Z_TRUTH[:,4], Z_TRUTH[:,5], 
                GLOBAL_Z_HIT[:,0], GLOBAL_Z_HIT[:,1], GLOBAL_Z_HIT[:,2], GLOBAL_Z_HIT[:,3], GLOBAL_Z_HIT[:,4], GLOBAL_Z_HIT[:,5],
                FIT_Z_HIT[:,0], FIT_Z_HIT[:,1], FIT_Z_HIT[:,2], FIT_Z_HIT[:,3], FIT_Z_HIT[:,4], FIT_Z_HIT[:,5],  
    ]

    df = pd.DataFrame(data=np.column_stack(df_data), columns=df_columns)
    df.dropna(subset=["GLOBAL_Z_HIT_6"], inplace=True, ignore_index=True)
    df.to_csv(output_path)

    return df

def GenerateACTSData(df, outdir, nmuon = True):

    # Define the particle data types
    particles_dtype = {
        "event_id": np.int32,
        "particle_id": "unsigned long",
        "particle_type": "int",
        "process": "unsigned int",
        "vx": "double",
        "vy": "double",
        "vz": "double",
        "vt": "double",
        "px": "double",
        "py": "double",
        "pz": "double",
        "m": "double",
        "q": "double",
        "eta": "double",
        "phi": "double",
        "theta": "double",
        "pt": "double",
        "p": "double",
        "vertex_primary": "unsigned int",
        "vertex_secondary": "unsigned int",
        "particle": "unsigned int",
        "generation": "unsigned int",
        "sub_particle": "unsigned int"
    }

    # Create a ROOT file and tree
    output = r.TFile(outdir, "RECREATE")
    tree = r.TTree("particles", "Particle information")

    # Create vectors for each branch
    event_id = np.zeros(1, dtype=np.int32)
    particle_id = r.vector('unsigned long')()
    particle_type = r.vector('int')()
    process = r.vector('unsigned int')()
    vx = r.vector('double')()
    vy = r.vector('double')()
    vz = r.vector('double')()
    vt = r.vector('double')()
    px = r.vector('double')()
    py = r.vector('double')()
    pz = r.vector('double')()
    m = r.vector('double')()
    q = r.vector('double')()
    eta = r.vector('double')()
    phi = r.vector('double')()
    theta = r.vector('double')()
    pt = r.vector('double')()
    p = r.vector('double')()
    vertex_primary = r.vector('unsigned int')()
    vertex_secondary = r.vector('unsigned int')()
    particle = r.vector('unsigned int')()
    generation = r.vector('unsigned int')()
    sub_particle = r.vector('unsigned int')()

    # Add the branches to the tree
    tree.Branch("event_id", event_id,"event_id/i")
    tree.Branch("particle_id", particle_id)
    tree.Branch("particle_type", particle_type)
    tree.Branch("process", process)
    tree.Branch("vx", vx)
    tree.Branch("vy", vy)
    tree.Branch("vz", vz)
    tree.Branch("vt", vt)
    tree.Branch("px", px)
    tree.Branch("py", py)
    tree.Branch("pz", pz)
    tree.Branch("m", m)
    tree.Branch("q", q)
    tree.Branch("eta", eta)
    tree.Branch("phi", phi)
    tree.Branch("theta", theta)
    tree.Branch("pt", pt)
    tree.Branch("p", p)
    tree.Branch("vertex_primary", vertex_primary)
    tree.Branch("vertex_secondary", vertex_secondary)
    tree.Branch("particle", particle)
    tree.Branch("generation", generation)
    tree.Branch("sub_particle", sub_particle)

    # Retrieve the data from the muon simulation
    vx_mu = df["GLOBAL_X_HIT_1"]
    vy_mu = df["GLOBAL_Y_HIT_1"]
    vz_mu = df["GLOBAL_Z_HIT_1"]
    px_mu = df["FIT_PX_1"]
    py_mu = df["FIT_PY_1"]
    pz_mu = df["PZ_FIT"]
    p_mu = df["P_FIT"]
    eta_mu = df["ETA_FIT"]
    theta_mu = df["THETA_FIT"]
    phi_mu = df["PHI_FIT"]
    pt_mu = df["PT_FIT"]

    if nmuon == True:
        particle_type_value = 13
        q_value = -1
    else:
        particle_type_value = -13
        q_value = 1

    n_list_muon=len(px_mu)

    for i in range(n_list_muon):
        event_id[0] = i
        particle_id.push_back(4503599644147712)
        particle_type.push_back(particle_type_value)  # Assign 13 or -13 depending on the charge
        process.push_back(0)
        vx.push_back(vx_mu[i])        
        vy.push_back(vy_mu[i])       #vertex_y'>vertex_x'->-vertex_z  IMPORTANT2 and IMPORTANT1
        vz.push_back(vz_mu[i])        #vertex_z' ->-vertex_x-> vertex_y  *IMPORTANT1 and IMPORTANT2  
        vt.push_back(0)               # vt is not specified in the muon data, fill it with some value
        px.push_back(px_mu[i])        #pz'->px
        py.push_back(py_mu[i])       #py'->px'->-pz
        pz.push_back(pz_mu[i])        #pz'->-px->py
        m.push_back(0.1056583745)     # mass of muon in GeV
        q.push_back(q_value) # charge of muon
        eta.push_back(eta_mu[i])
        phi.push_back(phi_mu[i])
        theta.push_back(theta_mu[i])
        pt.push_back(pt_mu[i])
        p.push_back(p_mu[i])
        vertex_primary.push_back(1)
        vertex_secondary.push_back(0)
        particle.push_back(1)
        generation.push_back(0)
        sub_particle.push_back(0)

        tree.Fill()

        particle_id.clear()
        particle_type.clear()
        process.clear()
        vx.clear()
        vy.clear()
        vz.clear()
        vt.clear()
        px.clear()
        py.clear()
        pz.clear()
        m.clear()
        q.clear()
        eta.clear()
        phi.clear()
        theta.clear()
        pt.clear()
        p.clear()
        vertex_primary.clear()
        vertex_secondary.clear()
        particle.clear()
        generation.clear()
        sub_particle.clear()

    # Write the ROOT tree to the output file and close it
    output.Write()
    output.Close()

    return 0

def Recreate_Input(input_dir, i, outdir):

    df = Generate_DataFrame_From_ROOT(input_dir, i)
    GenerateACTSData(df, outdir)


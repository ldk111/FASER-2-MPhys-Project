import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT as r

inputdir = "nmuon_acts.root"
outdir = "nmuon_acts_sample.root"
sample = 50000

def GenerateACTSData(inputdir, outdir, sample = 0, nmuon = True):

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

    #Take information from the muon simulation
    filename_mu=inputdir
    file_mu = uproot.open(filename_mu)
    tree_mu = file_mu["particles"] 

    if sample == 0:

        # Retrieve the data from the muon simulation
        vx_mu = tree_mu["vx"].array()
        vy_mu = tree_mu["vy"].array()
        vz_mu = tree_mu["vz"].array()
        px_mu = tree_mu["px"].array()
        py_mu = tree_mu["py"].array()
        pz_mu = tree_mu["pz"].array()
        p_mu = tree_mu["p"].array()
        eta_mu = tree_mu["eta"].array()
        theta_mu = tree_mu["theta"].array()
        phi_mu = tree_mu["phi"].array()
        pt_mu = tree_mu["pt"].array()

    else:

        # Retrieve the data from the muon simulation
        vx_mu = tree_mu["vx"].array()
        index = np.random.randint(0, len(vx_mu), sample)
        vx_mu = vx_mu[index]
        vy_mu = tree_mu["vy"].array()[index]
        vz_mu = tree_mu["vz"].array()[index]
        px_mu = tree_mu["px"].array()[index]
        py_mu = tree_mu["py"].array()[index] 
        pz_mu = tree_mu["pz"].array()[index]
        p_mu = tree_mu["p"].array()[index]
        eta_mu = tree_mu["eta"].array()[index]
        theta_mu = tree_mu["theta"].array()[index]
        phi_mu = tree_mu["phi"].array()[index]
        pt_mu = tree_mu["pt"].array()[index]

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

GenerateACTSData(inputdir, outdir, sample=sample)
import sys
sys.path.insert(0, "../pythonPackages")

import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT as r

inputdir = "pmuon_filter_final.root"
outdir ="pmuon_filter_200k_"
nmuon=False
sample = 200000

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

    #Take information from the muon simulation
    filename_mu=inputdir
    file_mu = uproot.open(filename_mu)
    tree_mu = file_mu["particles"]

    for i in range(0, len(tree_mu["vx"].array())//sample):

        # Create a ROOT file and tree
        output = r.TFile(outdir + str(i) + ".root", "RECREATE")
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
        vx_mu = tree_mu["vx"].array()[i*sample:(i+1)*sample]
        vy_mu = tree_mu["vy"].array()[i*sample:(i+1)*sample]
        vz_mu = tree_mu["vz"].array()[i*sample:(i+1)*sample]
        px_mu = tree_mu["px"].array()[i*sample:(i+1)*sample]
        py_mu = tree_mu["py"].array()[i*sample:(i+1)*sample] 
        pz_mu = tree_mu["pz"].array()[i*sample:(i+1)*sample]
        p_mu = tree_mu["p"].array()[i*sample:(i+1)*sample]
        eta_mu = tree_mu["eta"].array()[i*sample:(i+1)*sample]
        theta_mu = tree_mu["theta"].array()[i*sample:(i+1)*sample]
        phi_mu = tree_mu["phi"].array()[i*sample:(i+1)*sample]
        pt_mu = tree_mu["pt"].array()[i*sample:(i+1)*sample]

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
            vx.push_back(vx_mu[i][0])        
            vy.push_back(vy_mu[i][0])       #vertex_y'>vertex_x'->-vertex_z  IMPORTANT2 and IMPORTANT1
            vz.push_back(vz_mu[i][0])        #vertex_z' ->-vertex_x-> vertex_y  *IMPORTANT1 and IMPORTANT2  
            vt.push_back(0)               # vt is not specified in the muon data, fill it with some value
            px.push_back(px_mu[i][0])        #pz'->px
            py.push_back(py_mu[i][0])       #py'->px'->-pz
            pz.push_back(pz_mu[i][0])        #pz'->-px->py
            m.push_back(0.1056583745)     # mass of muon in GeV
            q.push_back(q_value) # charge of muon
            eta.push_back(eta_mu[i][0])
            phi.push_back(phi_mu[i][0])
            theta.push_back(theta_mu[i][0])
            pt.push_back(pt_mu[i][0])
            p.push_back(p_mu[i][0])
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

GenerateACTSData(inputdir, outdir, sample, nmuon)
import ROOT as r
import uproot3 as uproot
import pyhepmc.io
import numpy as np
import random


output_file = f'/data/atlassmallfiles/users/salin/Acts_x/GEN/Muon_FLArE/Test/Particles_muon_FLArE_test.root'


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
output = r.TFile(output_file, "RECREATE")
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


# Calculate the particle's eta, phi, pt,p
#IMPORTANT1 : There was a rotation in the Axis from XYZ into Z'Y'X'(new beamline along X' and Magnetic field along Z') to make ACTS work Z-> X' and X->-Z'
#IMPORTANT2 Different definition ACTS and FORESEE Y->X->Z'

#Take information from the muon simulation
filename_mu='D:\\ARPE\\FLArE\\muons.root'
file_mu = uproot.open(filename_mu)
tree_mu = file_mu["muons"] 

# Retrieve the data from the muon simulation
vx_mu = tree_mu.array("vz") #vertex_x' -> vertex_z *IMPORTANT1
vy_mu = - tree_mu.array("vx") #vertex_y'>vertex_x'->-vertex_z  IMPORTANT2 and IMPORTANT1
vz_mu = tree_mu.array("vy") #vertex_z' ->-vertex_x-> vertex_y  *IMPORTANT1 and IMPORTANT2 
px_mu = tree_mu.array("pz") #pz'->px
py_mu = - tree_mu.array("px") #py'->px'->-pz -> -px
pz_mu = tree_mu.array("py") #pz'->-px->py
E_mu = tree_mu.array("E")

p_mu=np.sqrt(px_mu**2+py_mu**2+pz_mu**2)

theta_mu, phi_mu, eta_mu, pt_mu = np.arccos(pz_mu/p_mu), np.arctan2(py_mu, px_mu), 0.5 * np.log((p_mu + pz_mu) / (p_mu - pz_mu)), np.sqrt(px_mu**2 + py_mu**2)
n_list_muon=len(px_mu)

for i in range(n_list_muon):
    event_id[0] = i
    particle_id.push_back(4503599644147712)
    particle_type.push_back(13)  # Assign 13 or -13 depending on the charge
    process.push_back(0)
    vx.push_back(vx_mu[i])        
    vy.push_back(vy_mu[i])       #vertex_y'>vertex_x'->-vertex_z  IMPORTANT2 and IMPORTANT1
    vz.push_back(vz_mu[i])        #vertex_z' ->-vertex_x-> vertex_y  *IMPORTANT1 and IMPORTANT2  
    vt.push_back(0)               # vt is not specified in the muon data, fill it with some value
    px.push_back(px_mu[i])        #pz'->px
    py.push_back(py_mu[i])       #py'->px'->-pz
    pz.push_back(pz_mu[i])        #pz'->-px->py
    m.push_back(0.1056583745)     # mass of muon in GeV
    q.push_back(1) # charge of muon
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
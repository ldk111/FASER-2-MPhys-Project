import ROOT as r

import pyhepmc.io
import numpy as np
import random

z_FLArE_F2= 10260
output_file = f'/data/atlassmallfiles/users/salin/acts_F2/GEN/Muon_FLArE/convert/test/Particles_muon_FLArE_test_z{z_FLArE_F2}m.root'


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
# Load muon data
filename_mu = '/data/atlassmallfiles/users/salin/acts_F2/GEN/Muon_FLArE/muons.root'
file_mu = r.TFile(filename_mu)
tree_mu = file_mu.Get("muons")



n_entries = tree_mu.GetEntries()

for i in range(n_entries):
    tree_mu.GetEntry(i)

    vx_mu = tree_mu.vz # IMPORTANT1: vertex_x' -> vertex_z
    vy_mu = - tree_mu.vx # IMPORTANT1 and IMPORTANT2: vertex_y'>vertex_x'->-vertex_z-> -vertex_x
    vz_mu = tree_mu.vy #vertex_z' ->-vertex_x-> vertex_y  *IMPORTANT1 and IMPORTANT2
    px_mu = tree_mu.pz #pz'->px
    py_mu = - tree_mu.px #py'->px'->-pz -> -px
    pz_mu = tree_mu.py #pz'->-px->py
    E_mu = tree_mu.E


    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    theta_mu = np.arccos(pz_mu/p_mu)
    phi_mu = np.arctan2(py_mu, px_mu)
    eta_mu = 0.5 * np.log((p_mu + pz_mu) / (p_mu - pz_mu))
    pt_mu = np.sqrt(px_mu**2 + py_mu**2)

    event_id[0] = i
    particle_id.push_back(4503599660924928)
    particle_type.push_back(13)  # assuming muon if pz<0, else anti-muon
    process.push_back(0)
    vx.push_back(vx_mu-z_FLArE_F2)  # IMPORTANT1: vertex_x' -> vertex_z
    vy.push_back(vy_mu)  # IMPORTANT1 and IMPORTANT2: vertex_y'>vertex_x'->-vertex_z
    vz.push_back(vz_mu)  # IMPORTANT1 and IMPORTANT2: vertex_z' ->-vertex_x-> vertex_y
    vt.push_back(0)
    px.push_back(px_mu)  # pz'->px
    py.push_back(py_mu)  # py'->px'->-pz
    pz.push_back(pz_mu)  # pz'->-px->py
    m.push_back(0.105658)  # muon mass
    q.push_back(-1)  # assuming muon if pz<0, else anti-muon
    eta.push_back(eta_mu)
    phi.push_back(phi_mu)
    theta.push_back(theta_mu)
    pt.push_back(pt_mu)
    p.push_back(p_mu)
    vertex_primary.push_back(1)
    vertex_secondary.push_back(0)
    particle.push_back(1)  # assuming muon if pz<0, else anti-muon
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

output.Write()
output.Close()

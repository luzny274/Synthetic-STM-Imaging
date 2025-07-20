import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import os.path
import matplotlib
from math import sqrt, pi
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from os.path import isfile, join

import faulthandler
faulthandler.enable()

import png
from PIL import Image

from os import listdir
from os.path import isfile, join

###################
##Lattice definition
def get_lattice_definition():
    name_file_str = open("get_simulation_name.bat", "r").read()
    
    name = name_file_str[name_file_str.find("=") + 1:]
    print("name: " + name)
    a = 0.332

    a1     = np.array([a, 0, 0])
    a2     = np.array([a/2, a/2 * np.sqrt(3), 0])
    sub_S1 = np.array([0, 0, 0])
    sub_S2 = np.array([0, 0, -a])
    sub_W  = np.array([0, a/np.sqrt(3), -a/2])

    sublattices = {'S1' : sub_S1, 'S2' : sub_S2, 'W' : sub_W}
    t0=1.19

    lat = pb.Lattice(a1, a2)
    lat.add_sublattices(('Se1', sub_S1),
                        ('Se2', sub_S2),
                        ('W', sub_W))
    lat.add_hoppings(
        #inside the main cell
        ([0,  0], 'Se1', 'W', t0),
        ([0,  0], 'Se2', 'W', t0),

        #outside the main cell
        ([1, -1], 'Se2', 'W', t0),
        ([1, -1], 'Se1', 'W', t0),
        ([0, -1], 'Se2', 'W', t0),
        ([0, -1], 'Se1', 'W', t0),
    )   
    return lat, sublattices, a, a1, a2, name
############################################################################

###################
##Lattice definition
def get_lattice_parameters():
    name_file_str = open("get_simulation_name.bat", "r").read()
    
    name = name_file_str[name_file_str.find("=") + 1:]
    print("name: " + name)
    a = 0.332

    a1     = np.array([a, 0, 0])
    a2     = np.array([a/2, a/2 * np.sqrt(3), 0])
    sub_S1 = np.array([0, 0, 0])
    sub_S2 = np.array([0, 0, -a])
    sub_W  = np.array([0, a/np.sqrt(3), -a/2])

    sublattices = {'S1' : sub_S1, 'S2' : sub_S2, 'W' : sub_W}
    return sublattices, a, a1, a2, name

def urand(val_range):
    return np.random.uniform() * (val_range[1] - val_range[0]) + val_range[0]

def get_range(params, key):
    return [params[f"min_{key}"], params[f"max_{key}"]]

###################
##LDOS model
def get_sample_size(params):
    sample_size = urand(get_range(params, "Sample_Size"))
    l_LDOS  = sample_size + 4
    l_model = l_LDOS + 30
    return sample_size, l_model, l_LDOS

def get_energy_of_interest(params): return np.random.choice(params["E_oi_sign_arr"]) * urand(get_range(params, "E_oi_value")) #Energy of interest cannot exceed 1 eV!
def get_resolution(params):         return params["Resolution"]
def get_z(params):                  return urand(get_range(params, "Initial_Z"))
def get_kappa(params):              return urand(get_range(params, "Kappa"))

############################################
#############Piezo parameters

def get_piezo_constant_drift_f(params):    return urand(get_range(params, "Constant_Drift_f"))
def get_piezo_constant_drift_s(params):    return urand(get_range(params, "Constant_Drift_s"))
def get_piezo_random_drift_f(params):      return urand(get_range(params, "Random_Offset_f"))
def get_piezo_random_drift_s(params):      return urand(get_range(params, "Random_Offset_s"))
def get_piezo_creep_params_f(params):      return urand(get_range(params, "Creep_Gamma_f")), urand(get_range(params, "Creep_Magnitude_f"))
def get_piezo_creep_params_s(params):      return urand(get_range(params, "Creep_Gamma_s")), urand(get_range(params, "Creep_Magnitude_s"))
def get_piezo_hysteresis_params_z(params): return urand(get_range(params, "Hysteresis_A_z")), urand(get_range(params, "Hysteresis_Beta_z")), urand(get_range(params, "Hysteresis_Gamma_z"))
def get_gauss_noise_parameters(params):    return urand(get_range(params, "Guassian_Noise_Amplitude"))


###################
##Defects
def get_defect_parameters(l_LDOS, params, sublattices):
    sites = list(sublattices.keys())

    n_vac = (np.array([urand(get_range(params, site + "_Vacancies_per_nm2")) for site in sites]) * (l_LDOS ** 2)).astype(int)
    n_dop = (np.array([urand(get_range(params, site + "_Dopants_per_nm2")) for site in sites]) * (l_LDOS ** 2)).astype(int)

    dop_V = [np.random.uniform(params["min_" + sites[i] + "_Dopant_Potential"], params["max_" + sites[i] + "_Dopant_Potential"], size = (n_dop[i])) * np.random.choice(params[sites[i] + "_Dopant_Potential_sign_arr"], size = (n_dop[i])) for i in range(n_dop.shape[0])]
    dop_V = np.concatenate(dop_V)
    return n_vac, n_dop, dop_V

def get_all_tip_shapes(folder):
    return [f for f in listdir(folder) if ".npy" in f]

def get_rotation_matrix(angle, axis):
    K = np.array([[ 0      , -axis[2],  axis[1]], 
                    [ axis[2], 0       , -axis[0]], 
                    [-axis[1], axis[0] , 0       ]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def rotate(matrix, array):
    flat = array.reshape(-1, 3)
    rotated = (matrix @ flat.T).T
    return rotated.reshape(array.shape)

###################
##Tip shape
def get_tip_shape(a, params):
    tip_shape_selection = params["select_Tip_Shape"]

    if tip_shape_selection == "Random": #Random selection
        tip_files = get_all_tip_shapes("tip_ldos")
        tip_file = "tip_ldos/" + tip_files[np.random.randint(len(tip_files))]
    else:
        tip_file = "tip_ldos/" + tip_shape_selection
    ldos_tip = np.load(tip_file)

    tilt_sz = urand(get_range(params, "Tip_Tilt"))
    angle_tilt = urand([0.0, 2*np.pi])
    tilt_axis = [np.sin(angle_tilt), np.cos(angle_tilt), 0.0]

    angle = urand(get_range(params, "Tip_Rotation"))

    tilt_rot = get_rotation_matrix(tilt_sz, tilt_axis)
    tip_rot = get_rotation_matrix(angle, [0.0, 0.0, 1.0])

    ldos_tip[:, :, :3] = rotate(tilt_rot, rotate(tip_rot, ldos_tip[:, :, :3]))

    ldos_tip[:, :, 0] -= np.mean(ldos_tip[:, :, 0])
    ldos_tip[:, :, 1] -= np.mean(ldos_tip[:, :, 1])
    ldos_tip[:, :, 2] -= np.min (ldos_tip[:, :, 2])
    return ldos_tip

def get_tilt(params):
    axis_angle = np.random.uniform(0.0, 2 * np.pi)
    tilt_axis = [np.cos(axis_angle), np.sin(axis_angle), 0.0]
    tilt_angle = urand(get_range(params, "Sample_Tilt"))

    return tilt_angle, tilt_axis

def get_rotation(params):       return urand(get_range(params, "Sample_Rotation"))
def get_scanning_dim(params):   return np.random.choice(params["Fast_Axis_Dimension_arr"])
def get_f_scanning_dir(params): return np.random.choice(params["Fast_Scanning_Direction_arr"])
def get_s_scanning_dir(params): return np.random.choice(params["Slow_Scanning_Direction_arr"])

def get_PID(params): return urand(get_range(params, "K_P")), urand(get_range(params, "K_D"))


############################################################################
##Basic functions
def rectangle(width, height):
    x0 = width / 2
    y0 = height / 2
    return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])

def vacancy(positions, radius):
    @pb.site_state_modifier
    def modifier(state, x, y, z):
        for i in range(positions.shape[0]):
            pos = positions[i, :]
            state[(x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2 < radius**2] = False
        return state
    return modifier

def dopant(positions, V):
    @pb.onsite_energy_modifier
    def potential(x, y, z):
        pot = np.zeros_like(x)
        for i in range(positions.shape[0]):
            pos = positions[i, :]
            pot[(x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2 < (1e-2)**2] = V[i]
        return pot
    return potential

def calculate_sample_ldos(n_vac, n_dop, dop_V, l_model, l_LDOS, lat, sublattices, a, a1, a2, E_range, E_reso, gamma):  
    side_atom_countx = l_LDOS // np.max(a1) // 2
    side_atom_county = l_LDOS // np.max(a2) // 2
    sublattices_offsets = np.array(list(sublattices.values()))
    def random_positions(ns):
        n = np.sum(ns)
        
        x = np.random.randint(-side_atom_countx, side_atom_countx, size = (n))
        y = np.random.randint(-side_atom_county, side_atom_county, size = (n))

        sites = []
        for i in range(ns.shape[0]) : sites += [i] * ns[i] 
        sites = np.array(sites)
        
        return (x[:, None] * a1[None, :] + y[:, None] * a2[None, :] + sublattices_offsets[sites], sites)
    
    vac_positions, site_vacancies = np.array([])[:, None], np.array([])
    dop_positions, site_dopants   = np.array([])[:, None], np.array([])

    if np.sum(n_vac) > 0 : vac_positions, site_vacancies = random_positions(n_vac)
    if np.sum(n_dop) > 0 : dop_positions, site_dopants   = random_positions(n_dop)  

    # side_atom_countx = l_LDOS // np.max(a1) // 2
    # side_atom_county = l_LDOS // np.max(a2) // 2
    # def random_positions(ns):
    #     n = np.sum(ns)
        
    #     x = np.random.randint(-side_atom_countx, side_atom_countx, size = (n))
    #     y = np.random.randint(-side_atom_county, side_atom_county, size = (n))
    
    #     sites = np.array([0] * ns[0] + [1] * ns[1] + [2] * ns[2])
        
    #     return (x[:, None] * a1[None, :] + y[:, None] * a2[None, :] + sublattices[sites], sites)
    
    # vac_positions, site_vacancies = np.array([])[:, None], np.array([])
    # dop_positions, site_dopants   = np.array([])[:, None], np.array([])
    # if np.sum(n_vac) > 0 : vac_positions, site_vacancies = random_positions(n_vac)
    # if np.sum(n_dop) > 0 : dop_positions, site_dopants   = random_positions(n_dop)
    
    ###################
    ##LDOS calculation
    model = pb.Model(
        lat, 
        rectangle(l_model,l_model),
        vacancy(vac_positions, radius=0.1),
        dopant(dop_positions, V = dop_V)
    )
    kpm = pb.kpm(model)
    energies = np.linspace(-E_range, E_range, E_reso)
    spatial_ldos = kpm.calc_spatial_ldos(energy=energies, broadening=gamma, shape=rectangle(l_LDOS,l_LDOS))
    
    num_of_sample_atoms = spatial_ldos.structure_map(0).positions.x.shape[0]
    ldos = np.zeros((energies.shape[0], num_of_sample_atoms, 4))
    i=0
    for energy in energies:
        smap = spatial_ldos.structure_map(energy)
        ldos_data = np.array(smap.data,ndmin=1)
        ldos_positions = np.array(smap.positions).T
        ldos[i, :, 0:3] = ldos_positions[:, 0:3]
        ldos[i, :, 3] = ldos_data
        i += 1

    return ldos, vac_positions, site_vacancies, dop_positions, site_dopants

def calculate_ldos_sums(ldos_sample, ldos_tip, E_range, E_reso, E_oi):
    ###################
    ##LDOS summation
    if np.abs(E_oi) > E_range:
        print("Error: Energy of interest cannot exceed energy range!")
    
    num_of_tip_atoms    = ldos_tip.shape[1]
    
    i_e0 = int(ldos_sample.shape[0] / 2)
    i_eV = int((E_oi * E_reso) / (2 * E_range))

    if E_oi > 0:
        ldos_sum = ldos_sample[(i_e0 - i_eV):i_e0, : , 3][:, :, None] * ldos_tip[i_e0:(i_e0 + i_eV), :, 3][:, None, :]
    else:
        ldos_sum = ldos_sample[i_e0:(i_e0 - i_eV), : , 3][:, :, None] * ldos_tip[(i_e0 + i_eV):i_e0, :, 3][:, None, :]
    ldos_sum = np.sum(ldos_sum, axis = 0)
    
    return ldos_sum
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

import ctypes
ctypes.CDLL("libiomp5md.dll", mode=ctypes.RTLD_GLOBAL)
import torch

from Sim_Params import *
from PiezoElectricActuator import *

def init_pytorch(use_cuda):
    d_dtype = torch.float32
    torch.set_default_dtype(d_dtype)

    # Initialize PyTorch
    cuda_available = torch.cuda.is_available()
    USE_CUDA = use_cuda and cuda_available
    print("Cuda available: ", cuda_available)
    print("Using Cuda:     ", USE_CUDA)
    if USE_CUDA:
        print(torch.cuda.get_device_name(0))
        
    device = 'cuda' if USE_CUDA else 'cpu'
    torch.set_default_device(device)
    return device, d_dtype

##Calculating currents
#global stability_constant
stability_constant = None
def calc_currents(ldos_in_fov, kappa, dists_xy2, zs_plane, s_atoms, t_atoms, d_dtype, stability_constant = None):
    dists_z = (zs_plane[:, :, None, None] + t_atoms[:, 2][None, None, :, None] - s_atoms[:, 2][None, None, None, :])
    dists = torch.sqrt(dists_xy2 + dists_z ** 2)
    
    if stability_constant is None:
        stability_constant = float(2 * kappa * torch.min(dists))

    decays = torch.exp(- 2 * kappa * dists + stability_constant)

    eps = 1e-12
    
    currents = torch.sum(decays * ldos_in_fov.T[None, None, :, :], axis = (2, 3), dtype=d_dtype) + eps

    currents_ln = torch.log(currents)

    derivatives = -decays * ldos_in_fov.T[None, None, :, :] * 2 * kappa * dists_z / dists
    derivative = torch.sum(derivatives, axis = (2, 3))

    derivative /= currents
    
    return currents_ln, derivative, stability_constant

def transform_and_crop_positions(positions, rotation, translation, thr, vals = None):
    pos = (rotation @ positions.T).T + translation
    cond = (pos[:, 0] <= thr) & (pos[:, 0] >= -thr) & (pos[:, 1] <= thr) & (pos[:, 1] >= -thr)
    pos_in_fov = pos[cond]
    
    vals_in_fov = vals
    if vals is not None:
        vals_in_fov = vals[cond]
        
    return pos, pos_in_fov, vals_in_fov

#Newton method for calculating the height profile
def calc_height_profile(pxls, z, ldos_in_fov, kappa, dists_xy2, steps, atom_in_fov, t_atoms, d_dtype):
    #Initial z values (flat)
    zs_plane = torch.zeros((pxls, pxls), dtype = d_dtype) + z
    
    currents_original, derivative, stability_constant = calc_currents(ldos_in_fov, kappa, dists_xy2, zs_plane, atom_in_fov, t_atoms, d_dtype)

    currents = currents_original.detach().clone()
    target_current = torch.median(currents)
    errors = [torch.max(torch.abs(currents - target_current)).cpu()]

    eps = 1
    for i in range(steps):
        #Clamp derivative to improve stability
        derivative_clamped = derivative
        derivative_clamped[(derivative_clamped > 0) & (derivative_clamped <  eps)] =  eps
        derivative_clamped[(derivative_clamped < 0) & (derivative_clamped > -eps)] = -eps
        
        zs_plane += (target_current - currents) / derivative_clamped
        currents, derivative, _ = calc_currents(ldos_in_fov, kappa, dists_xy2, zs_plane, atom_in_fov, t_atoms, d_dtype, stability_constant)
        errors.append(float(torch.max(torch.abs(target_current - currents)).cpu()))

    error_relative = errors[-1] / errors[0]
    return currents_original, zs_plane, errors, error_relative


def generate_sample(params):
    sample_size, l_model, l_LDOS = get_sample_size(params)
    lat, sublattices, a, a1, a2, name = get_lattice_definition()
    n_vac, n_dop, dop_V = get_defect_parameters(l_LDOS, params, sublattices)
    ldos_tip = get_tip_shape(a, params)
    tilt_angle, tilt_axis = get_tilt(params)
    gauss_amp = get_gauss_noise_parameters(params)

    A, beta, gamma = get_piezo_hysteresis_params_z(params)
    K_P, K_D = get_PID(params)

    pxls  = get_resolution(params)
    z     = get_z(params)
    kappa = get_kappa(params)
    E_oi  = get_energy_of_interest(params)

    theta = get_rotation(params)
    

    f_dimension = get_scanning_dim(params)
    scanning_s_direction = get_s_scanning_dir(params)
    scanning_f_direction = get_f_scanning_dir(params)

    piezo_random_drift_f = get_piezo_random_drift_f(params)
    piezo_random_drift_s = get_piezo_random_drift_s(params)

    piezo_constant_drift_f = get_piezo_constant_drift_f(params)
    piezo_constant_drift_s = get_piezo_constant_drift_s(params)

    gamma_f, magnitude_f = get_piezo_creep_params_f(params)
    gamma_s, magnitude_s = get_piezo_creep_params_s(params)
    

    def np_to_t(arr):
        return torch.from_numpy(arr).to(dtype = d_dtype).to(device)
    

    E_range = 1
    E_reso = 200
    gamma = 0.1
    
    ldos_sample, vac_positions, site_vacancies, dop_positions, site_dopants = calculate_sample_ldos(n_vac, n_dop, dop_V, l_model, l_LDOS, lat, sublattices, a, a1, a2, E_range, E_reso, gamma)
    ldos_sum   = calculate_ldos_sums(ldos_sample, ldos_tip, E_range, E_reso, E_oi)

    s_atoms = ldos_sample[0, :, :-1]
    s_atoms_original = np.copy(s_atoms)
    s_atoms = (get_rotation_matrix(tilt_angle, tilt_axis) @ s_atoms.T).T

    #Tip ground truth
    
    a_tungsten = 0.3165
    max_tip_l = np.sqrt(3) * a_tungsten

    ldos_tip_relative = (get_rotation_matrix(-tilt_angle, tilt_axis) @ ldos_tip[0, :, :-1].T).T

    x_grid = np.linspace(-max_tip_l, max_tip_l, pxls)
    y_grid = np.linspace(-max_tip_l, max_tip_l, pxls)
    tip_dists_x = x_grid[:, None, None] - ldos_tip_relative[:, 0][None, None, :]
    tip_dists_y = y_grid[None, :, None] - ldos_tip_relative[:, 1][None, None, :]
    tip_dists_z = z - ldos_tip_relative[:, 2][None, None, :]

    tip_dists = np.sum(np.exp( - 2 * kappa * np.sqrt(tip_dists_x ** 2 + tip_dists_y ** 2 + tip_dists_z ** 2)), axis=2)
    tip_ground_truth = (tip_dists - tip_dists.min()) / (tip_dists.max() / tip_dists.min())

    device, d_dtype = init_pytorch(params["select_Backend"] == "GPU")

    rotation = np_to_t(np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta),  np.cos(theta), 0],
                                 [0            ,  0            , 1]]))
    translation = np_to_t(np.array([np.random.uniform(-a/2, a/2), np.random.uniform(-a/2, a/2), 0.0]))

    s_atoms_t   = np_to_t(s_atoms)
    t_atoms     = np_to_t(ldos_tip[0, :, :3])
    ldos_sum_t  = np_to_t(ldos_sum)

    atom_pos, atom_in_fov, ldos_in_fov = transform_and_crop_positions(s_atoms_t, rotation, translation, l_LDOS, vals = ldos_sum_t)

    ##Thermal drift
    f_random_offsets = torch.normal(torch.zeros((pxls)), piezo_random_drift_f)
    s_random_offsets = torch.normal(torch.zeros((pxls)), piezo_random_drift_s)

    f_drift_offsets = torch.linspace(0.0, 1.0, pxls, dtype = d_dtype) * piezo_constant_drift_f
    s_drift_offsets = torch.linspace(0.0, 1.0, pxls, dtype = d_dtype) * piezo_constant_drift_s

    ##Creep
    f_creep_offsets = gamma_f * torch.log(torch.linspace(1.0 / magnitude_f, 1.0, pxls, dtype = d_dtype))
    s_creep_offsets = gamma_s * torch.log(torch.linspace(1.0 / magnitude_s, 1.0, pxls, dtype = d_dtype))

    #f_creep_offsets -= torch.min(f_creep_offsets)
    #s_creep_offsets -= torch.min(s_creep_offsets)

    if scanning_s_direction: s_creep_offsets = -s_creep_offsets.flip(0)
    if scanning_f_direction: f_creep_offsets = -f_creep_offsets.flip(0)

    ##Grid
    f_grid = torch.linspace(-sample_size/2, sample_size/2, pxls, dtype=d_dtype)[:, None] + (f_drift_offsets + f_random_offsets + f_creep_offsets)[None, :]
    s_grid = torch.linspace(-sample_size/2, sample_size/2, pxls, dtype=d_dtype)[None, :] + (s_drift_offsets + s_random_offsets + s_creep_offsets)[None, :]

    f_grid -= torch.mean(f_grid)
    s_grid -= torch.mean(s_grid)

    dists_fs2  = (f_grid[:, :, None, None] + t_atoms[:, 0][None, None, :, None] - atom_in_fov[:, 0][None, None, None, :]) ** 2 + \
                 (s_grid[:, :, None, None] + t_atoms[:, 1][None, None, :, None] - atom_in_fov[:, 1][None, None, None, :]) ** 2
    zs_plane = torch.zeros((pxls, pxls), dtype = d_dtype) + z
    
    newton_steps = 10
    original_currents, heights, error_clean, error_relative = calc_height_profile(pxls, z, ldos_in_fov, kappa, dists_fs2 , newton_steps, atom_in_fov, t_atoms, d_dtype)
    print("Newton error (relative):", float(error_relative.cpu()))

    ##PID Circuit and Piezo Actuator simulation
    ind = dists_fs2.shape[0] - 1 if scanning_f_direction else 0
    row_dists = dists_fs2[ind:ind+1, :]
    
    row_zs0 = heights[ind:ind+1, :]
    row_zs = row_zs0.clone()
    zs = []
    zs.append(row_zs)

    currents, derivative, stability_constant = calc_currents(ldos_in_fov, kappa, row_dists, row_zs, atom_in_fov, t_atoms, d_dtype)

    prev_currents = currents.detach().clone()
    orig_currents_std = torch.std(currents)
    target_current = torch.median(currents)

    ideal_PI = torch.median(1.0 / derivative)

    zs_range = torch.max(heights) - torch.min(heights)
    zs_half_point = (torch.max(heights) + torch.min(heights)) / 2

    actuator = PiezoElectricActuator(A, beta, gamma, row_zs0 / zs_range, params["plot"] == "True")

    K_P_ratio = K_P
    K_P = K_P_ratio * ideal_PI

    start_time = time.time()

    inds = list(range(1, dists_fs2.shape[0]))
    if scanning_f_direction: 
        inds.reverse()
        inds = np.array(inds) - 1

    for ind in inds:
        row_zs = row_zs.detach().clone()
        e = currents - target_current
        de = currents - prev_currents

        z_movement = - e * K_P - de * K_D
        z_movement[(torch.abs(z_movement) > zs_range/2)] = torch.sign(z_movement[(torch.abs(z_movement) > zs_range/2)]) * zs_range/2
        
        row_zs = actuator.move(z_movement / zs_range, 1.0) * zs_range# + row_zs0

        row_dists = dists_fs2[ind:ind+1, :]
        prev_currents = currents.detach().clone()
        currents, derivative, stability_constant = calc_currents(ldos_in_fov, kappa, row_dists, row_zs, atom_in_fov, t_atoms, d_dtype, stability_constant)

        currents += torch.normal(torch.zeros_like(currents), torch.zeros_like(currents) + gauss_amp) * orig_currents_std

        zs.append(row_zs)

    if scanning_f_direction: zs.reverse()
    zs_final = torch.concatenate(zs, dim=0)

    ##Ground truth generation
    vac_positions_t = np_to_t(vac_positions)
    dop_positions_t = np_to_t(dop_positions)

    v_pos, v_pos_in_fov = vac_positions_t, vac_positions_t
    d_pos, d_pos_in_fov = dop_positions_t, dop_positions_t

    if vac_positions_t.shape[0] > 0: v_pos, v_pos_in_fov, _ = transform_and_crop_positions(vac_positions_t, rotation, translation, sample_size / 2)
    if dop_positions_t.shape[0] > 0: d_pos, d_pos_in_fov, _ = transform_and_crop_positions(dop_positions_t, rotation, translation, sample_size / 2)

    pois = pois = torch.concatenate((atom_in_fov, v_pos)) if v_pos.shape[0] > 0 else atom_in_fov

    dists_atoms  = (f_grid[:, :, None, None] - pois[:, 0][None, None, None, :]) ** 2 + (s_grid[:, :, None, None] - pois[:, 1][None, None, None, :]) ** 2 + pois[:, 2][None, None, None, :] ** 2
    
    def get_lattice_img(dists_atoms):
        dists = torch.sqrt(dists_atoms)
        lattice_img = 1.1 - torch.min(dists[:, :, 0, :], dim=2)[0] / (a / 2)
        lattice_img[lattice_img < 0] = 0
        lattice_img[lattice_img > 1] = 1
        return lattice_img
    lattice_img = get_lattice_img(dists_atoms)

    ##Defects img
    defects_img = torch.zeros_like(lattice_img)
    if v_pos.shape[0] > 0 or d_pos.shape[0] > 0:
        if v_pos.shape[0] > 0: pois = v_pos
        if d_pos.shape[0] > 0: pois = d_pos
        if v_pos.shape[0] > 0 and d_pos.shape[0] > 0: pois = torch.concatenate((d_pos, v_pos), dim = 0)
        dists_defects  = (f_grid[:, :, None, None] - pois[:, 0][None, None, None, :]) ** 2 + (s_grid[:, :, None, None] - pois[:, 1][None, None, None, :]) ** 2
        defects_img        = get_lattice_img(dists_defects)

    ground_truth = torch.concatenate((lattice_img[:, :, None], defects_img[:, :, None],torch.zeros_like(lattice_img)[:, :, None]), axis = 2)

    current_of_constant_height =      original_currents if not f_dimension else original_currents.T
    height_of_constant_current =                heights if not f_dimension else heights.T
    ground_truth               =           ground_truth if not f_dimension else torch.transpose(ground_truth, 0, 1)
    ldos_tip[:, :, [0, 1]]     = ldos_tip[:, :, [0, 1]] if not f_dimension else ldos_tip[:, :, [1, 0]]
    tip_ground_truth           =       tip_ground_truth if not f_dimension else tip_ground_truth.T

    actuator_vs = actuator.vs[::10, 0, :] if not f_dimension else actuator.vs[::10, 0, :].T
    zs_final    = zs_final                if not f_dimension else zs_final.T

    if scanning_f_direction: actuator_vs = actuator_vs.flip(f_dimension)

    return a, s_atoms, vac_positions, dop_positions, ldos_tip, ldos_sample, ldos_sum, tip_ground_truth, current_of_constant_height, height_of_constant_current, zs_final, actuator_vs, actuator, ground_truth, K_P_ratio, K_D, sample_size, l_model, l_LDOS, rotation, translation, error_relative, v_pos_in_fov.cpu().numpy(), d_pos_in_fov.cpu().numpy()

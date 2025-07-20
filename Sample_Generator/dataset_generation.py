import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import os.path
import matplotlib
from math import sqrt, pi
import os

from os.path import isfile, join

import faulthandler
faulthandler.enable()

import png
from PIL import Image

import Sim_Params

import argparse

import json

import Simulation as sim

import torch

param_folder = "gui_params/"
param_file = "main_params"
lat, sublattices, a, a1, a2, name = Sim_Params.get_lattice_definition()

def load_params():
    loaded_params = None
    try:
        with open(param_folder + param_file + ".json", "r") as file:
            loaded_params = json.load(file)
    except FileNotFoundError:
        print("Error: JSON file not found. Using default values.")
    return loaded_params

params = load_params()

parser = argparse.ArgumentParser("")
parser.add_argument("sample_count", help="Samples to be generated", type=int)
args = parser.parse_args()

sample_directory = name + "_dataset"
os.makedirs(sample_directory, exist_ok=True)

sample_files = [f for f in os.listdir(sample_directory) if isfile(join(sample_directory, f))]
if len(sample_files) > 0:
    sample_indices = [int(f[:f.find("_")]) for f in sample_files]
    index_0 = np.max(sample_indices)
else:
    index_0 = -1
print("Last generated sample index: ", index_0)

pxls = params["Resolution"]
params["plot"] = "False"

begin = time.time()

sample_index = index_0 + 1
while sample_index < args.sample_count + index_0 + 1:
    print("sample index: {}".format(sample_index))

    a, s_atoms, vac_positions, dop_positions, ldos_tip, ldos_sample, ldos_sum, tip_ground_truth, current_of_constant_height, height_of_constant_current, zs_final, actuator_vs, actuator, ground_truth, K_P, K_D, sample_size, l_model, l_LDOS, rotation, translation, error_relative, vac_pos_fov, dop_pos_fov = sim.generate_sample(params)
    if error_relative > 1e-2 or np.isnan(error_relative):
        print("SKIPPING SAMPLE; too high newton method error or NaN")
        continue

    def save_as_img(arr, file_name):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        if isinstance(arr, np.ndarray):
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = (arr.cpu().numpy() * 255).astype(np.uint8)
        

        with open(file_name + '.png', 'wb') as f:
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
                writer = png.Writer(width=pxls, height=pxls, bitdepth=8, greyscale=False)
                writer.write(f, arr)
            else:
                writer = png.Writer(width=arr.shape[1], height=arr.shape[0], bitdepth=8, greyscale=True)
                writer.write(f, arr.tolist())

    file_prefix = sample_directory + "/" + str(sample_index) + "_"

    save_as_img(tip_ground_truth, file_prefix + "tip_distribution")
    save_as_img(ground_truth    , file_prefix + "ground_truth")

    save_as_img(actuator_vs               , file_prefix + "voltage_profile")
    save_as_img(height_of_constant_current, file_prefix + "constant_current_profile")

    np.savez(file_prefix + "params.npz", defect_count   = len(vac_pos_fov) + len(dop_pos_fov),
                                         vacancy_count  = len(vac_pos_fov),
                                         dopant_count   = len(dop_pos_fov),
                                         vacancy_pos    = vac_pos_fov,
                                         dopant_pos     = dop_pos_fov,
                                         K_P            = K_P,
                                         K_D            = K_D,
                                         sample_size    = sample_size,
                                         tip_atom_count = ldos_tip.shape[1]
                                        )

    tot_time = time.time() - begin
    print("total time: {:.1f}s; avg time per sample: {:.2f}s".format(tot_time, tot_time / (sample_index - index_0)))
    
    sample_index += 1

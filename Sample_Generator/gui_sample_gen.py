import streamlit as st
import io
import json

import numpy as np
import matplotlib.pyplot as plt

import Simulation as sim
import Sim_Params

st.set_page_config(layout="wide")

param_folder = "gui_params/"

# Initialize session state
if "param_file" not in st.session_state:
    st.session_state.param_file = "default_params"
if "parameters" not in st.session_state:
    st.session_state.parameters = {}

st.session_state.param_file = st.sidebar.text_input("Parameter file", value="default_params")
st.session_state.constant_vals = st.sidebar.checkbox("Constant values")

sublattices, a, a1, a2, name = Sim_Params.get_lattice_parameters()

def get_param(key, default=0.0):
    return st.session_state.parameters.get(key, default)

def set_param(key, value):
    st.session_state.parameters[key] = value

def load_params():
    try:
        with open(param_folder + st.session_state.param_file + ".json", "r") as file:
            loaded_params = json.load(file)

            # Force Streamlit to register changes by assigning keys individually
            for key, value in loaded_params.items():
                st.session_state.parameters[key] = value
                st.session_state[key] = value  # Also update direct session state

    except FileNotFoundError:
        print("Error: JSON file not found. Using default values.")

def save_params():
    with open(param_folder + st.session_state.param_file + ".json", "w") as file:
        json.dump(st.session_state.parameters, file, indent=4)
        print("Parameters saved!")

def input_float_range(text):
    min_key, max_key = f"min_{text}", f"max_{text}"
    val_key = f"val_{text}"

    step = 1e-6
    number_format = "%.6f"

    if st.session_state.constant_vals:
        col0, col1 = st.sidebar.columns(2)
        with col0:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            st.text(text)
        with col1:
            value = st.number_input("val", step=step, key=val_key, format=number_format)
            set_param(val_key, st.session_state[val_key])
            set_param(min_key, st.session_state[val_key])
            set_param(max_key, st.session_state[val_key])
            value_min = value
            value_max = value
    else:
        col0, col1, col2 = st.sidebar.columns(3)
        with col0:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            st.text(text)
        with col1:
            value_min = st.number_input("min:", step=step, key=min_key, format=number_format)
            set_param(min_key, st.session_state[min_key])
        with col2:
            value_max = st.number_input("max:", step=step, key=max_key, format=number_format)
            set_param(max_key, st.session_state[max_key])

    return value_min, value_max

def input_selectbox(text, options):
    key = f"select_{text}"
    if key not in st.session_state: st.session_state[key] = options[0] 
    selected_value = st.sidebar.selectbox(text, options, key=key)
    set_param(key, st.session_state[key])
    return selected_value

def input_int(text, min_value=0, max_value=2048, step=1):
    key = text
    if key not in st.session_state: st.session_state[key] = min_value
    value = st.sidebar.number_input(text, min_value=min_value, max_value=max_value, step=step, key=key, format="%d")
    set_param(key, st.session_state[key])
    return value


# UI for saving/loading parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Save Parameters"):
        save_params()
with col2:
    if st.button("Load Parameters"):
        load_params()
        st.rerun()  # Ensure UI updates properly

st.session_state.parameters["plot"] = "True"

##########################################################################
##Rendering
if st.button("Render"):
    def urand(val_range):
        return np.random.uniform() * (val_range[1] - val_range[0]) + val_range[0]

    def get_range(params, key):
        return [params[f"min_{key}"], params[f"max_{key}"]]
    print(st.session_state.parameters)
    print("sample_size test: ", get_range(st.session_state.parameters, "Sample_Size"))

    a, s_atoms, vac_positions, dop_positions, ldos_tip, ldos_sample, ldos_sum, tip_ground_truth, current_of_constant_height, height_of_constant_current, zs_final, actuator_vs, actuator, ground_truth, K_P_ratio, K_D, sample_size, l_model, l_LDOS, rotation, translation, error_relative, v_pos_in_fov, d_pos_in_fov = sim.generate_sample(st.session_state.parameters)

    st.session_state.fig1, ax = plt.subplots(figsize=(10, 5))  
    ax.set_title("Sample Model")
    ax.set_aspect('equal')
    ax.scatter(s_atoms[:, 0], s_atoms[:, 1], ldos_sum[:, 0])
    if vac_positions.shape[0] > 0: ax.scatter(vac_positions[:, 0], vac_positions[:, 1], c='r')
    if dop_positions.shape[0] > 0: ax.scatter(dop_positions[:, 0], dop_positions[:, 1], c='g')

    st.session_state.buffer1 = io.BytesIO()
    st.session_state.fig1.savefig(st.session_state.buffer1, format="png", dpi=100, bbox_inches="tight")

    a_tungsten = 0.3165
    max_tip_l = np.sqrt(3) * a_tungsten

    st.session_state.fig2, ax = plt.subplots(figsize=(10, 5))  
    ax.set_title("Tip Atoms")
    ax.set_xlim([-max_tip_l, max_tip_l])
    ax.set_ylim([-max_tip_l, max_tip_l])
    ax.set_aspect('equal')

    ldos_sum = np.sum(ldos_tip[:, :, 3], axis=0)

    sc = ax.scatter(ldos_tip[0, :, 1], -ldos_tip[0, :, 0], c=ldos_sum)
    st.session_state.fig2.colorbar(sc, ax=ax)
    
    st.session_state.buffer2 = io.BytesIO()
    st.session_state.fig2.savefig(st.session_state.buffer2, format="png", dpi=100, bbox_inches="tight")

    def my_imshow(title, arr, cb_title):
        fig, ax = plt.subplots(figsize=(10, 5))  
        if title is not None:
            ax.set_title(title)
        im = ax.imshow((arr - arr.min())/(arr.max() - arr.min()), cmap="afmhot")
        ax.axis('off')
        buffer = io.BytesIO()
        if cb_title is not None:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cb_title, fontsize=14)
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        return buffer

    st.session_state.buffer3 = my_imshow("Tip Atoms' effect distribution", tip_ground_truth, None)
    st.session_state.buffer4 = my_imshow("Constant height mode", current_of_constant_height.cpu(), None)
    st.session_state.buffer5 = my_imshow("Constant current mode", height_of_constant_current.cpu(), None)
    st.session_state.buffer6 = my_imshow("PID sim; Actuator voltages", actuator_vs.cpu(), 'Voltage (arbitrary units)')
    st.session_state.buffer7 = my_imshow("PID sim; Tip Z", zs_final.cpu(), None)


    st.session_state.fig8, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("PID simulation; Actuator Hysteresis curve")
    ind = actuator.vs.shape[2] // 2

    print("actuator.vs.shape", actuator.vs.shape)

    vs = actuator.vs[:, 0, ind].cpu()
    vs = (vs - vs.min()) / (vs.max() - vs.min())

    xs = actuator.xs[:, 0, ind].cpu()
    xs = (xs - xs.min()) / (xs.max() - xs.min())

    sc = ax.scatter(vs, xs, c=actuator.ts.cpu(), cmap='plasma', marker='o')
    cbar = st.session_state.fig8.colorbar(sc)
    cbar.set_label('Time (arbitrary units)', fontsize=14)

    ax.set_aspect('equal')
    ax.set_xlabel('Voltage (arbitrary units)', fontsize=14)
    ax.set_ylabel('Displacement (arbitrary units)', fontsize=14)
    ax.set_title('Bouc-Wen Model cross-section', fontsize=14)
    ax.grid()

    st.session_state.buffer8 = io.BytesIO()
    st.session_state.fig8.savefig(st.session_state.buffer8, format="png", dpi=100, bbox_inches="tight")

    st.session_state.fig9, ax = plt.subplots(figsize=(10, 5))  
    ax.set_title("Ground Truth")
    ax.imshow(ground_truth.cpu())
    
    st.session_state.buffer9 = io.BytesIO()
    st.session_state.fig9.savefig(st.session_state.buffer9, format="png", dpi=100, bbox_inches="tight")


general_sim_params = ["E_oi_value", "Sample_Size", "Initial_Z", "Kappa", "Sample_Rotation", "Sample_Tilt"]

piezo_fast_axis_sim_params = ["Constant_Drift_f", "Random_Offset_f", "Creep_Gamma_f", "Creep_Magnitude_f"]
piezo_slow_axis_sim_params = ["Constant_Drift_s", "Random_Offset_s", "Creep_Gamma_s", "Creep_Magnitude_s"]

piezo_Z_axis_sim_params = ["Hysteresis_A_z", "Hysteresis_Beta_z", "Hysteresis_Gamma_z"]

sites = list(sublattices.keys())

defects_sim_params = [site + "_Vacancies_per_nm2" for site in sites] + [site + "_Dopants_per_nm2" for site in sites] + [site + "_Dopant_Potential" for site in sites]

def sign_selection(text):
    selected_value = input_selectbox(text + "_sign", ["+", "-", "+/-"])

    if selected_value == "+"  : set_param(text + "_sign_arr", [ 1])
    if selected_value == "-"  : set_param(text + "_sign_arr", [-1])
    if selected_value == "+/-": set_param(text + "_sign_arr", [ 1, -1])


st.sidebar.markdown("<p style='font-size:24px;'>General parameters</p>", unsafe_allow_html=True)
selected_value = input_selectbox( "Backend", ["CPU", "GPU"])

selected_value = input_selectbox("E_oi_sign", ["+", "-", "+/-"])
if selected_value == "+"  : set_param("E_oi_sign_arr", [ 1])
if selected_value == "-"  : set_param("E_oi_sign_arr", [-1])
if selected_value == "+/-": set_param("E_oi_sign_arr", [ 1, -1])
for param in general_sim_params: input_float_range(param)

selected_value = input_selectbox( "Fast_Axis_Dimension", ["X", "Y", "X/Y"])
if selected_value == "X"  : set_param("Fast_Axis_Dimension_arr", [0])
if selected_value == "Y"  : set_param("Fast_Axis_Dimension_arr", [1])
if selected_value == "X/Y": set_param("Fast_Axis_Dimension_arr", [0, 1])

selected_value = input_selectbox( "Fast_Scanning_Direction", ["->", "<-", "->/<-"])
if selected_value == "->"   : set_param("Fast_Scanning_Direction_arr", [0])
if selected_value == "<-"   : set_param("Fast_Scanning_Direction_arr", [1])
if selected_value == "->/<-": set_param("Fast_Scanning_Direction_arr", [0, 1])

selected_value = input_selectbox( "Slow_Scanning_Direction", ["->", "<-", "->/<-"])
if selected_value == "->"   : set_param("Slow_Scanning_Direction_arr", [0])
if selected_value == "<-"   : set_param("Slow_Scanning_Direction_arr", [1])
if selected_value == "->/<-": set_param("Slow_Scanning_Direction_arr", [0, 1])

input_int("Resolution")

st.sidebar.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:24px;'>Piezo Electric Actuators</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:16px; font-weight:bold;'>Fast Axis</p>", unsafe_allow_html=True)
for param in piezo_fast_axis_sim_params: input_float_range(param)
st.sidebar.markdown("<p style='font-size:16px; font-weight:bold;'>Slow Axis</p>", unsafe_allow_html=True)
for param in piezo_slow_axis_sim_params: input_float_range(param)
st.sidebar.markdown("<p style='font-size:16px; font-weight:bold;'>Z Axis</p>", unsafe_allow_html=True)
for param in piezo_Z_axis_sim_params: input_float_range(param)

st.sidebar.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:24px;'>Noise</p>", unsafe_allow_html=True)
input_float_range("Guassian_Noise_Amplitude")

st.sidebar.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:24px;'>Defects</p>", unsafe_allow_html=True)
for param in defects_sim_params: input_float_range(param)
[sign_selection(site + "_Dopant_Potential") for site in sites]

st.sidebar.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:24px;'>Tip</p>", unsafe_allow_html=True)
tip_options = ["Random"] + Sim_Params.get_all_tip_shapes("tip_ldos")
selected_value = input_selectbox("Tip_Shape", tip_options)

input_float_range("Tip_Rotation")
input_float_range("Tip_Tilt")

st.sidebar.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:24px;'>PID Circuit</p>", unsafe_allow_html=True)
input_float_range("K_P")
input_float_range("K_D")

st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

if "fig1" in st.session_state:
    col0, col1, col2 = st.columns(3)
    with col0:
        st.image(st.session_state.buffer1)
    with col1:
        st.image(st.session_state.buffer2)
    with col2:
        st.image(st.session_state.buffer3)
        
    col0, col1, col2 = st.columns(3)
    with col0:
        st.image(st.session_state.buffer4)
    with col1:
        st.image(st.session_state.buffer5)
    with col2:
        st.image(st.session_state.buffer9)
        
    col0, col1, col2 = st.columns(3)
    with col0:
        st.image(st.session_state.buffer6)
    with col1:
        st.image(st.session_state.buffer7)
    with col2:
        st.image(st.session_state.buffer8)
        
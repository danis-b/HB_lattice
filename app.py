import streamlit as st
from HB_lattice import HB_lattice
import matplotlib.pyplot as plt
import numpy as np

# Main app layout
st.title("HB Lattice Simulator")
st.write("Welcome! Follow the steps: Create a lattice, plot the Hofstadter butterfly, then plot DOS or spatial maps.")

# Sidebar: Parallel option
st.sidebar.header("Simulation Settings")
parallel = st.sidebar.checkbox("Run in Parallel", value=False, help="Enable parallel processing for eigenvalue calculations.")

# Initialize session state
if 'lattice_created' not in st.session_state:
    st.session_state.lattice_created = False
if 'hofstadter_plotted' not in st.session_state:
    st.session_state.hofstadter_plotted = False
if 'latt' not in st.session_state:
    st.session_state.latt = HB_lattice(parallel=parallel)

# Reference to the lattice object
latt = st.session_state.latt

# Sync parallel setting with checkbox
latt.parallel = parallel

# Sidebar for lattice creation
st.sidebar.header("Lattice Parameters")
latt_type = st.sidebar.selectbox("Lattice Type", ["square", "triangular", "honeycomb"])
num_cells = st.sidebar.slider("Number of Cells", 1, 20, 1)
bond_len = st.sidebar.number_input("Bond Length (nm)", min_value=0.1, value=10.0)

if st.sidebar.button("Create New Lattice"):
    # Reset the lattice object with current parallel setting
    st.session_state.latt = HB_lattice(parallel=parallel)
    latt = st.session_state.latt
    latt.create_lattice(latt_type, num_cells, bond_len)
    st.session_state.lattice_created = True
    st.session_state.hofstadter_plotted = False  # Reset Hofstadter state
    st.pyplot(plt.gcf())  # Display the lattice plot
    st.write(f"New {latt_type} lattice created with {num_cells} cells and bond length {bond_len} nm (Parallel: {parallel}).")

# Hofstadter plot section
st.sidebar.header("Hofstadter Plot Parameters")
b_max = st.sidebar.number_input("Max Magnetic Field (T)", min_value=0.1, value=41.3)
b_steps = st.sidebar.slider("Steps", 10, 500, 200)
g_factor = st.sidebar.number_input("g-factor", value=0.0)
ham_type = st.sidebar.selectbox("Hamiltonian Type", ["hopping", "interpolation"])

# Dynamic Hamiltonian parameters
if ham_type == "hopping":
    t_input = st.sidebar.text_input("Hopping values (eV)", value="-0.1", help="Comma-separated real numbers, e.g., 0.1, 0.05")
    t_so_input = st.sidebar.text_input("Spin-Orbit hopping (eV)", value="0", help="Comma-separated complex numbers, e.g., 0.01, 0.005j")
    
    try:
        t_values = [x.strip() for x in t_input.split(',') if x.strip()]
        if not t_values:
            raise ValueError("No valid values provided")
        t = [float(x) for x in t_values]
    except ValueError as e:
        st.sidebar.error(f"Invalid input for t: {e}. Use comma-separated real numbers (e.g., 1.0, 0.5).")
        t = [0.1]

    try:
        t_so_values = [x.strip() for x in t_so_input.split(',') if x.strip()]
        if not t_so_values:
            raise ValueError("No valid values provided")
        t_so = [complex(x) for x in t_so_values]
    except ValueError as e:
        st.sidebar.error(f"Invalid input for t_so: {e}. Use comma-separated (complex) numbers (e.g., 0.01, 0.005j).")
        t_so = [0.0]
        
    ham_params = {"t": t, "t_so": t_so}
    st.write("Hamiltonian parameters:", ham_params)

elif ham_type == "interpolation":
    a_param = st.sidebar.number_input("a_param (eV)", value=1.0, step=0.1)
    b_param = st.sidebar.number_input("b_param (nm)", value=0.5, step=0.1)
    ham_params = {"a_param": a_param, "b_param": b_param}

if st.sidebar.button("Plot/Replot Hofstadter", disabled=not st.session_state.lattice_created):
    latt.parallel = parallel  # Update parallel setting
    latt.plot_hofstadter(b_max, b_steps, g_factor, ham_type, **ham_params)
    st.pyplot(plt.gcf())
    st.write(f"Hofstadter plot generated with B_max={b_max} T, {b_steps} steps, g={g_factor}, {ham_type} Hamiltonian (Parallel: {parallel}).")
    st.session_state.hofstadter_plotted = True

# DOS plot section
if st.session_state.hofstadter_plotted:
    st.sidebar.header("DOS Plot Parameters")
    b_value_input = st.sidebar.text_input("B values for DOS (T)", value="0, 20.5", help="Comma-separated, e.g., 0,10,20")
    e_min = st.sidebar.number_input("Energy Min (eV)", value=-0.25)
    e_max = st.sidebar.number_input("Energy Max (eV)", value=0.25)
    e_step = st.sidebar.number_input("Energy Steps", min_value=100, value=1000, step=100)
    smear = st.sidebar.number_input("Smearing (eV)", value=0.001, step=0.0001, format="%.4f")
    try:
        b_value = [float(x.strip()) for x in b_value_input.split(',')]
    except ValueError:
        st.sidebar.error("Invalid B values. Use comma-separated numbers (e.g., 0,10,20).")
        b_value = [0.0]

    if st.sidebar.button("Plot DOS"):
        latt.plot_dos(b_value, e_min, e_max, e_step, smear)
        st.pyplot(plt.gcf())
        st.write(f"DOS plotted for B={b_value} T, energy range {e_min} to {e_max} eV, {e_step} steps, smear={smear} eV.")

# Spatial map plot section
if st.session_state.hofstadter_plotted:
    st.sidebar.header("Spatial Map Parameters")
    b_value_map = st.sidebar.number_input("B value for map (T)", min_value=0.0, max_value=b_max, value=0.0, step=0.1)
    num_eigvecs_input = st.sidebar.text_input("Eigenvector Indices", value="0,1", help="Comma-separated, e.g., 0,1")
    mapRes = st.sidebar.slider("Map Resolution", 50, 200, 100)
    smear_map = st.sidebar.number_input("Smearing (nm)", value=10.0, step=1.0)
    try:
        num_eigvecs = [int(x.strip()) for x in num_eigvecs_input.split(',')]
    except ValueError:
        st.sidebar.error("Invalid eigenvector indices. Use comma-separated integers (e.g., 0,1).")
        num_eigvecs = [0]

    if st.sidebar.button("Plot Spatial Map"):
        latt.plot_map(b_value_map, num_eigvecs, mapRes, smear_map)
        st.pyplot(plt.gcf())
        st.write(f"Spatial map plotted for B={b_value_map} T, eigenvectors {num_eigvecs}, resolution {mapRes}, smear={smear_map} nm.")

# Dynamic instructions
if not st.session_state.lattice_created:
    st.sidebar.write("Create a lattice to unlock Hofstadter plotting.")
elif not st.session_state.hofstadter_plotted:
    st.sidebar.write("Plot the Hofstadter butterfly to unlock DOS and spatial map plotting.")
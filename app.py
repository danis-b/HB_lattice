import streamlit as st
from HB_lattice import HB_lattice
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from io import BytesIO

# Main app layout
st.title("Hofstadter's Butterfly Simulator")
st.write("Create a lattice, plot the Hofstadter butterfly with DOS, and visualize spatial maps.")

# Initialize session state
if 'lattice_created' not in st.session_state:
    st.session_state.lattice_created = False
if 'hofstadter_plotted' not in st.session_state:
    st.session_state.hofstadter_plotted = False
if 'latt' not in st.session_state:
    st.session_state.latt = HB_lattice(parallel=True)  # Always parallel
if 'lattice_fig' not in st.session_state:
    st.session_state.lattice_fig = None
if 'hofstadter_fig' not in st.session_state:
    st.session_state.hofstadter_fig = None
if 'spatial_fig' not in st.session_state:
    st.session_state.spatial_fig = None
if 'combined_fig' not in st.session_state:
    st.session_state.combined_fig = None

# Reference to the lattice object
latt = st.session_state.latt

# Sidebar for lattice creation
st.sidebar.header("Lattice Parameters")
latt_type = st.sidebar.selectbox("Lattice Type", ["square", "triangular", "honeycomb", "lattice from file"])

# Conditionally show num_cells and bond_len inputs
if latt_type != "lattice from file":
    num_cells = st.sidebar.slider("Number of Cells (N)", 1, 10, 1)
    bond_len = st.sidebar.number_input("Bond Length (nm)", min_value=0.1, value=10.0, step=0.1)
else:
    num_cells = None
    bond_len = None

# Show file uploader if "lattice from file" is selected
uploaded_file = None
if latt_type == "lattice from file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Coordinates File (.csv, .txt, .dat)", 
        type=["csv", "txt", "dat"]
    )

if st.sidebar.button("Create New Lattice"):
    st.session_state.lattice_fig = None
    st.session_state.hofstadter_fig = None
    st.session_state.spatial_fig = None
    st.session_state.combined_fig = None
    
    st.session_state.latt = HB_lattice(parallel=True)  # Always parallel
    latt = st.session_state.latt
    if latt_type == "lattice from file":
        if uploaded_file is not None:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                latt.create_custom_lattice(temp_file_path)
                st.session_state.lattice_created = True
                st.session_state.hofstadter_plotted = False
                st.session_state.lattice_fig = plt.gcf()
                st.write(f"Custom lattice created from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            st.error("Please upload a file.")
    else:
        latt.create_lattice(latt_type, num_cells, bond_len)
        st.session_state.lattice_created = True
        st.session_state.hofstadter_plotted = False
        st.session_state.lattice_fig = plt.gcf()
        st.write(f"New {latt_type} lattice: {num_cells} cells, {bond_len} nm.")

# Hofstadter plot section
st.sidebar.header("Hofstadter Parameters")
b_max = st.sidebar.number_input("Max B (T)", min_value=0.1, value=41.3, step=0.1)
b_steps = st.sidebar.slider("Steps", 10, 300, 200)
g_factor = st.sidebar.number_input("g-factor", value=0.0, step=0.1)
ham_type = st.sidebar.selectbox("Hamiltonian Type", ["hopping", "interpolation"])

# Dynamic Hamiltonian parameters in two columns
if ham_type == "hopping":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        t_input = st.text_input("t (eV)", value="-0.1", help="e.g., -0.1, 0.05")
    with col2:
        t_so_input = st.text_input("t_so (eV)", value="0", help="e.g., 0.01, 0.005j")
    try:
        t = [float(x.strip()) for x in t_input.split(',') if x.strip()]
        if not t:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"t error: {e}")
        t = [-0.1]
    try:
        t_so = [complex(x.strip()) for x in t_so_input.split(',') if x.strip()]
        if not t_so:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"t_so error: {e}")
        t_so = [0.0]
    ham_params = {"t": t, "t_so": t_so}
elif ham_type == "interpolation":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a_param = st.number_input("a_param (eV)", value=1.0, step=0.1)
    with col2:
        b_param = st.number_input("b_param (nm)", value=0.5, step=0.1)
    ham_params = {"a_param": a_param, "b_param": b_param}

if st.sidebar.button("Plot Hofstadter", disabled=not st.session_state.lattice_created):
    try:
        latt.plot_hofstadter(b_max, b_steps, g_factor, ham_type, **ham_params)
        st.session_state.hofstadter_fig = plt.gcf()
        st.session_state.hofstadter_plotted = True
        st.session_state.combined_fig = None
        st.write(f"Hofstadter: B_max={b_max} T, {b_steps} steps, g={g_factor}.")
    except Exception as e:
        st.error(f"Error: {e}")

# DOS plot section
if st.session_state.hofstadter_plotted:
    st.sidebar.header("DOS Parameters")
    b_value_input = st.sidebar.text_input("B values (T)", value="0, 20.5")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        e_step = st.number_input("E Steps", min_value=100, value=1000, step=100)
    with col2:
        smear = st.number_input("Smear (eV)", value=0.001, step=0.0001, format="%.4f")
    try:
        b_value = [float(x.strip()) for x in b_value_input.split(',') if x.strip()]
        if not b_value:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"B error: {e}")
        b_value = [0.0]

    if st.sidebar.button("Plot DOS"):
        try:
            e_min = np.min(latt.set_eigvals) + 0.1 * np.min(latt.set_eigvals)
            e_max = np.max(latt.set_eigvals) + 0.1 * np.max(latt.set_eigvals)
            latt.plot_dos(b_value, e_min, e_max, e_step, smear)
            dos_fig = plt.gcf()
            combined_fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax1 = combined_fig.add_subplot(gs[0])
            ax2 = combined_fig.add_subplot(gs[1], sharey=ax1)
            color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            if st.session_state.hofstadter_fig is not None:
                hof_data = st.session_state.hofstadter_fig.get_axes()[0].get_lines()
                for line in hof_data:
                    ax1.plot(line.get_xdata(), line.get_ydata(), color="blue", linewidth=1)
                for i, b in enumerate(b_value):
                    ax1.axvline(x=b, linewidth=2, linestyle='dashed', color=color_cycle[i % len(color_cycle)])
                ax1.set_xlim(0, b_max)
                ax1.set_xlabel("B (T)")
                ax1.set_ylabel("E (eV)")
            dos_axes = dos_fig.get_axes()[0]
            dos_lines = dos_axes.get_lines()
            for i, line in enumerate(dos_lines):
                energy = line.get_xdata()
                dos = line.get_ydata()
                ax2.plot(dos, energy, linewidth=1, color=color_cycle[i % len(color_cycle)])
                ax2.set_ylim(e_min, e_max)
            ax2.set_xlabel("DOS")
            ax2.tick_params(axis='y', labelleft=False)
            combined_fig.tight_layout()
            st.session_state.combined_fig = combined_fig
            st.write(f"DOS: B={b_value} T, {e_step} steps, smear={smear} eV.")
        except ValueError as e:
            st.error(f"DOS error: {e}")

# Spatial map plot section
if st.session_state.hofstadter_plotted:
    st.sidebar.header("Spatial Map Parameters")
    b_value_map = st.sidebar.number_input("B (T)", min_value=0.0, max_value=b_max, value=0.0, step=0.1)
    num_eigvecs_input = st.sidebar.text_input("Eigvec Indices", value="0,1")
    mapRes = st.sidebar.slider("Resolution", 50, 200, 100)
    smear_map = st.sidebar.number_input("Smear (nm)", value=10.0, step=1.0)
    try:
        num_eigvecs = [int(x.strip()) for x in num_eigvecs_input.split(',') if x.strip()]
        if not num_eigvecs:
            raise ValueError("No valid indices")
    except ValueError as e:
        st.sidebar.error(f"Indices error: {e}")
        num_eigvecs = [0]

    if st.sidebar.button("Plot Spatial Map"):
        try:
            latt.plot_map(b_value_map, num_eigvecs, mapRes, smear_map)
            st.session_state.spatial_fig = plt.gcf()
            st.write(f"Map: B={b_value_map} T, eigvecs {num_eigvecs}.")
        except ValueError as e:
            st.error(f"Map error: {e}")

# Divide screen into three parts
col1, col2 = st.columns(2)
bottom_container = st.container()

with col1:
    if st.session_state.lattice_fig is not None:
        st.pyplot(st.session_state.lattice_fig)
        buf = BytesIO()
        st.session_state.lattice_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(label="save", data=buf, file_name="lattice_plot.png", mime="image/png")

with col2:
    if st.session_state.spatial_fig is not None:
        st.pyplot(st.session_state.spatial_fig)
        buf = BytesIO()
        st.session_state.spatial_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(label="save", data=buf, file_name="spatial_map.png", mime="image/png")

with bottom_container:
    if st.session_state.combined_fig is not None:
        st.pyplot(st.session_state.combined_fig)
        buf = BytesIO()
        st.session_state.combined_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(label="save", data=buf, file_name="hofstadter_dos_plot.png", mime="image/png")
    elif st.session_state.hofstadter_fig is not None:
        st.pyplot(st.session_state.hofstadter_fig)
        buf = BytesIO()
        st.session_state.hofstadter_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(label="save", data=buf, file_name="hofstadter_plot.png", mime="image/png")

# Dynamic instructions
if not st.session_state.lattice_created:
    st.sidebar.write("Select lattice type and create it.")
elif not st.session_state.hofstadter_plotted:
    st.sidebar.write("Plot Hofstadter to unlock DOS and map.")

# GitHub link
st.markdown(
    """
    <div style='text-align: center; position: fixed; bottom: 10px; width: 100%;'>
        <a href='https://github.com/danis-b/HB_lattice' target='_blank'>GitHub (danis-b)</a>
    </div>
    """,
    unsafe_allow_html=True
)
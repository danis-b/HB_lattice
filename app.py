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
    st.session_state.latt = HB_lattice(parallel=False)
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

# Sync parallel setting with checkbox
latt.parallel = st.sidebar.checkbox("Run in Parallel", value=False, help="Enable parallel processing (requires joblib).")

# Sidebar for lattice creation
st.sidebar.header("Lattice Parameters")
latt_type = st.sidebar.selectbox("Lattice Type", ["square", "triangular", "honeycomb", "lattice from file"])

# Conditionally show num_cells and bond_len inputs only if not 'lattice from file'
if latt_type != "lattice from file":
    num_cells = st.sidebar.slider("Number of Cells (N)", 1, 20, 1, help="N x N lattice; Not used for file.")
    bond_len = st.sidebar.number_input("Bond Length (nm)", min_value=0.1, value=10.0, help="Not used for file.")
else:
    num_cells = None
    bond_len = None

# Show file uploader if "lattice from file" is selected
uploaded_file = None
if latt_type == "lattice from file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Coordinates File (.csv, .txt, .dat)", 
        type=["csv", "txt", "dat"], 
        help="File must contain two columns (x, y) in nm."
    )

if st.sidebar.button("Create New Lattice"):
    # Clear all figures to start from scratch
    st.session_state.lattice_fig = None
    st.session_state.hofstadter_fig = None
    st.session_state.spatial_fig = None
    st.session_state.combined_fig = None
    
    # Proceed with creating the new lattice
    st.session_state.latt = HB_lattice(parallel=latt.parallel)
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
                st.write(f"Custom lattice created from {uploaded_file.name} (Parallel: {latt.parallel}).")
            except Exception as e:
                st.error(f"Error creating custom lattice: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            st.error("Please upload a file for custom lattice creation.")
    else:
        latt.create_lattice(latt_type, num_cells, bond_len)
        st.session_state.lattice_created = True
        st.session_state.hofstadter_plotted = False
        st.session_state.lattice_fig = plt.gcf()
        st.write(f"New {latt_type} lattice created with {num_cells} cells and bond length {bond_len} nm (Parallel: {latt.parallel}).")

# Hofstadter plot section
st.sidebar.header("Hofstadter Parameters")
b_max = st.sidebar.number_input("Max Magnetic Field (T)", min_value=0.1, value=41.3)
b_steps = st.sidebar.slider("Steps", 10, 500, 200)
g_factor = st.sidebar.number_input("g-factor", value=0.0)
ham_type = st.sidebar.selectbox("Hamiltonian Type", ["hopping", "interpolation"])

# Dynamic Hamiltonian parameters
if ham_type == "hopping":
    t_input = st.sidebar.text_input("Hopping values (eV)", value="-0.1", help="Comma-separated real numbers, e.g., -0.1, 0.05")
    t_so_input = st.sidebar.text_input("Spin-Orbit hopping (eV)", value="0", help="Comma-separated numbers, e.g., 0.01, 0.005j")
    try:
        t_values = [x.strip() for x in t_input.split(',') if x.strip()]
        if not t_values:
            raise ValueError("No valid values provided")
        t = [float(x) for x in t_values]
    except ValueError as e:
        st.sidebar.error(f"Invalid input for t: {e}. Use comma-separated real numbers (e.g., -0.1, 0.05).")
        t = [-0.1]
    try:
        t_so_values = [x.strip() for x in t_so_input.split(',') if x.strip()]
        if not t_so_values:
            raise ValueError("No valid values provided")
        t_so = [complex(x) for x in t_so_values]
    except ValueError as e:
        st.sidebar.error(f"Invalid input for t_so: {e}. Use comma-separated numbers (e.g., 0.01, 0.005j).")
        t_so = [0.0]
    ham_params = {"t": t, "t_so": t_so}
elif ham_type == "interpolation":
    a_param = st.sidebar.number_input("a_param (eV)", value=1.0, step=0.1)
    b_param = st.sidebar.number_input("b_param (nm)", value=0.5, step=0.1)
    ham_params = {"a_param": a_param, "b_param": b_param}

if st.sidebar.button("Plot Hofstadter", disabled=not st.session_state.lattice_created):
    latt.parallel = latt.parallel
    try:
        latt.plot_hofstadter(b_max, b_steps, g_factor, ham_type, **ham_params)
        st.session_state.hofstadter_fig = plt.gcf()
        st.session_state.hofstadter_plotted = True
        st.session_state.combined_fig = None  # Reset combined figure until DOS is plotted
        st.write(f"Hofstadter plot generated with B_max={b_max} T, {b_steps} steps, g={g_factor}, {ham_type} Hamiltonian (Parallel: {latt.parallel}).")
    except ImportError as e:
        st.error(f"Parallel mode requires joblib: {e}. Install with 'pip install joblib' or disable parallel.")
    except Exception as e:
        st.error(f"Error plotting Hofstadter: {e}")

# DOS plot section with concatenation
if st.session_state.hofstadter_plotted:
    st.sidebar.header("DOS Parameters")
    b_value_input = st.sidebar.text_input("B values for DOS (T)", value="0, 20.5", help="Comma-separated, e.g., 0,10,20")
    e_step = st.sidebar.number_input("Energy Steps", min_value=100, value=1000, step=100)
    smear = st.sidebar.number_input("Smearing (eV)", value=0.001, step=0.0001, format="%.4f")
    try:
        b_value = [float(x.strip()) for x in b_value_input.split(',') if x.strip()]
        if not b_value:
            raise ValueError("No valid B values provided")
    except ValueError as e:
        st.sidebar.error(f"Invalid B values: {e}. Use comma-separated numbers (e.g., 0,10,20).")
        b_value = [0.0]

    if st.sidebar.button("Plot DOS"):
        try:
            # Dynamically get energy range from Hofstadter plot
            if latt.set_eigvals is not None:
                e_min = np.min(latt.set_eigvals) + 0.1 * np.min(latt.set_eigvals)
                e_max = np.max(latt.set_eigvals) + 0.1 * np.max(latt.set_eigvals)
            else:
                e_min, e_max = -0.25, 0.25  # Fallback if eigenvalues aren’t available

            # Generate the DOS plot with the correct energy range
            latt.plot_dos(b_value, e_min, e_max, e_step, smear)
            dos_fig = plt.gcf()

            # Create a combined figure with a 3:1 width ratio
            combined_fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # Hofstadter 3x wider than DOS

            # Hofstadter subplot (left, wider)
            ax1 = combined_fig.add_subplot(gs[0])
            # DOS subplot (right, narrower, sharing y-axis with Hofstadter)
            ax2 = combined_fig.add_subplot(gs[1], sharey=ax1)

            color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            # Plot Hofstadter data on ax1
            if st.session_state.hofstadter_fig is not None:
                hof_data = st.session_state.hofstadter_fig.get_axes()[0].get_lines()
                for line in hof_data:
                    ax1.plot(line.get_xdata(), line.get_ydata(), color="blue", linewidth=1)
                
                for i, b in enumerate(b_value):
                    ax1.axvline(x = b, linewidth=2, linestyle='dashed', color=color_cycle[i % len(color_cycle)])
                ax1.set_xlim(0, b_max)
                ax1.set_xlabel("Magnetic field (T)")
                ax1.set_ylabel("Energy (eV)")

            # Plot DOS data on ax2 (rotated: DOS on x-axis, energy on y-axis)
            dos_axes = dos_fig.get_axes()[0]
            dos_lines = dos_axes.get_lines()
            for i, line in enumerate(dos_lines):
                energy = line.get_xdata()
                dos = line.get_ydata()
                ax2.plot(dos, energy, linewidth=1, color=color_cycle[i % len(color_cycle)])
                ax2.set_ylim(e_min, e_max)
            ax2.set_xlabel("DOS")
            ax2.tick_params(axis='y', labelleft=False)  # Hide y-axis labels on DOS (shared with Hofstadter)

            # Adjust layout to prevent overlap
            combined_fig.tight_layout()
            st.session_state.combined_fig = combined_fig
            st.write(f"DOS plotted for B={b_value} T, {e_step} steps, smear={smear} eV.")
        except ValueError as e:
            st.error(f"Error plotting DOS: {e}")

# Spatial map plot section
if st.session_state.hofstadter_plotted:
    st.sidebar.header("Spatial Map Parameters")
    b_value_map = st.sidebar.number_input("B value for map (T)", min_value=0.0, max_value=b_max, value=0.0, step=0.1)
    num_eigvecs_input = st.sidebar.text_input("Eigenvector Indices", value="0,1", help="Comma-separated, e.g., 0,1")
    mapRes = st.sidebar.slider("Map Resolution", 50, 200, 100)
    smear_map = st.sidebar.number_input("Smearing (nm)", value=10.0, step=1.0)
    try:
        num_eigvecs = [int(x.strip()) for x in num_eigvecs_input.split(',') if x.strip()]
        if not num_eigvecs:
            raise ValueError("No valid indices provided")
    except ValueError as e:
        st.sidebar.error(f"Invalid eigenvector indices: {e}. Use comma-separated integers (e.g., 0,1).")
        num_eigvecs = [0]

    if st.sidebar.button("Plot Spatial Map"):
        try:
            latt.plot_map(b_value_map, num_eigvecs, mapRes, smear_map)
            st.session_state.spatial_fig = plt.gcf()
            st.write(f"Spatial map plotted for B={b_value_map} T, eigenvectors {num_eigvecs}, resolution {mapRes}, smear={smear_map} nm.")
        except ValueError as e:
            st.error(f"Error plotting spatial map: {e}")

# Divide screen into three parts
col1, col2 = st.columns(2)
bottom_container = st.container()

# Part 1: Lattice (Top Left)
with col1:
    # st.subheader("Lattice")
    if st.session_state.lattice_fig is not None:
        st.pyplot(st.session_state.lattice_fig)
        # Download button for lattice figure
        buf = BytesIO()
        st.session_state.lattice_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(
            label="save plot",
            data=buf,
            file_name="lattice_plot.png",
            mime="image/png"
        )

# Part 2: Spatial Map (Top Right)
with col2:
    # st.subheader("Spatial Map")
    if st.session_state.spatial_fig is not None:
        st.pyplot(st.session_state.spatial_fig)
        # Download button for spatial map figure
        buf = BytesIO()
        st.session_state.spatial_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(
            label="save plot",
            data=buf,
            file_name="spatial_map.png",
            mime="image/png"
        )

# Part 3: Combined Hofstadter + DOS (Bottom)
with bottom_container:
    # st.subheader("Hofstadter Butterfly with DOS")
    if st.session_state.combined_fig is not None:
        st.pyplot(st.session_state.combined_fig)
        # Download button for combined figure
        buf = BytesIO()
        st.session_state.combined_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(
            label="save plot",
            data=buf,
            file_name="hofstadter_dos_plot.png",
            mime="image/png"
        )
    elif st.session_state.hofstadter_fig is not None:
        st.pyplot(st.session_state.hofstadter_fig)
        # Download button for Hofstadter figure
        buf = BytesIO()
        st.session_state.hofstadter_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button(
            label="save plot",
            data=buf,
            file_name="hofstadter_plot.png",
            mime="image/png"
        )

# Dynamic instructions
if not st.session_state.lattice_created:
    st.sidebar.write("Select a lattice type and create it to unlock Hofstadter plotting. For 'lattice from file', upload a file.")
elif not st.session_state.hofstadter_plotted:
    st.sidebar.write("Plot the Hofstadter to unlock DOS and spatial map plotting.")
    
    
# Add GitHub link at the bottom
st.markdown(
    """
    <div style='text-align: center; position: fixed; bottom: 10px; width: 100%;'>
        <a href='https://github.com/danis-b/HB_lattice' target='_blank'>View this project on GitHub (danis-b)</a>
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
from HB_lattice import HB_lattice
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from io import BytesIO
import logging
import gc

# Configure logging to record errors in 'app.log' for debugging purposes
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to generate text file content for data export
def generate_data_text(latt, b_values, dos_data=None):
    """Generate text content for eigenvalues and optional DOS data."""
    text = "# Eigenvalues\n# B (T)   eigenvalues (eV)\n"
    for b, eigvals in zip(b_values, latt.set_eigvals):
        eigvals_str = ", ".join(map(str, np.round(eigvals, 6)))
        text += f"{np.round(b, 4)},  {eigvals_str}\n"
    
    if dos_data:
        for b_val, energy, dos in dos_data:
            text += f"\n# DOS for B = {b_val}\n# Energy (eV)   DOS\n"
            for e, d in zip(energy, np.round(dos, 6)):
                text += f"{np.round(e, 4)}   {d}\n"
    
    return text

# Streamlit app setup
st.title("Hofstadter's Butterfly Simulator")
st.write("Create a lattice, plot the Hofstadter butterfly with DOS, and visualize spatial maps.")

# Initialize session state to manage app state
if 'lattice_created' not in st.session_state:
    st.session_state.lattice_created = False  # Tracks if lattice is created
    st.session_state.hofstadter_plotted = False  # Tracks if Hofstadter plot is generated
    st.session_state.latt = None  # Stores the lattice object
    st.session_state.lattice_fig = None  # Stores the lattice figure
    st.session_state.hofstadter_fig = None  # Stores the Hofstadter butterfly plot
    st.session_state.spatial_fig = None  # Stores the spatial map plot
    st.session_state.combined_fig = None  # Stores the combined Hofstadter and DOS plot
    st.session_state.b_max = None  # Maximum magnetic field
    st.session_state.b_steps = None  # Number of magnetic field steps
    st.session_state.b_values = None  # Array of magnetic field values
    st.session_state.dos_data = None  # DOS data if calculated

# Sidebar: Lattice Creation
st.sidebar.header("Lattice Parameters")
latt_type = st.sidebar.selectbox("Lattice Type", ["square", "triangular", "honeycomb", "lattice from file"])

# Show num_cells and bond_len for non-file lattice types
if latt_type != "lattice from file":
    num_cells = st.sidebar.slider("Number of Cells", 1, 10, 1)  # Limited to prevent memory issues
    bond_len = st.sidebar.number_input("Bond Length (nm)", min_value=0.1, value=10.0, step=0.1)  # Avoid zero/negative
else:
    num_cells = None
    bond_len = None

# File uploader for custom lattice
uploaded_file = None
if latt_type == "lattice from file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Coordinates File (.csv, .txt, .dat)", 
        type=["csv", "txt", "dat"]
    )

# Button to create lattice
if st.sidebar.button("Create New Lattice"):
    try:
        # Clear previous lattice data
        if st.session_state.latt is not None:
            del st.session_state.latt
            gc.collect()  # Free memory
        st.session_state.latt = HB_lattice(parallel=True)  # New lattice object
        latt = st.session_state.latt
        
        # Reset figures and parameters
        st.session_state.lattice_fig = None
        st.session_state.hofstadter_fig = None
        st.session_state.spatial_fig = None
        st.session_state.combined_fig = None
        st.session_state.b_max = None
        st.session_state.b_steps = None
        st.session_state.b_values = None
        st.session_state.dos_data = None
        
        # Create lattice based on type
        if latt_type == "lattice from file" and uploaded_file is not None:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save temp file
            try:
                st.session_state.lattice_fig = latt.create_custom_lattice(temp_file_path)
                st.session_state.lattice_created = True
                st.session_state.hofstadter_plotted = False
                st.write(f"Custom lattice created from {uploaded_file.name}.")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)  # Clean up
        elif latt_type != "lattice from file":
            st.session_state.lattice_fig = latt.create_lattice(latt_type, num_cells, bond_len)
            st.session_state.lattice_created = True
            st.session_state.hofstadter_plotted = False
            st.write(f"New {latt_type} lattice: {num_cells} cells, {bond_len} nm.")
        else:
            st.error("Please upload a file for custom lattice.")
    except Exception as e:
        st.error(f"Error creating lattice: {e}")
        logging.error(f"Error creating lattice: {e}", exc_info=True)

# Sidebar: Hofstadter Parameters
st.sidebar.header("Hofstadter Parameters")
b_max = st.sidebar.number_input("Max Magnetic Field (T)", min_value=0.1, value=41.3, step=0.1)  # Avoid invalid field
b_steps = st.sidebar.slider("Steps", 10, 300, 200)  # Reasonable computation range
g_factor = st.sidebar.number_input("g-factor", value=0.0, step=0.1)  # Default g-factor
ham_type = st.sidebar.selectbox("Hamiltonian Type", ["hopping", "interpolation"])  # Hamiltonian options

# Hamiltonian parameters based on type
if ham_type == "hopping":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        t_input = st.text_input("Hopping array (eV)", value="-0.1", help="e.g., -0.1, 0.05, 0.001 are nn, nnn and 3rd neighbor hoppings")
    with col2:
        t_so_input = st.text_input("Spin-orbit (eV)", value="0", help="e.g., 0.01, 0.005j, 0.0001 are nn, nnn and 3rd neighbor SO hoppings")
    try:
        t = [float(x.strip()) for x in t_input.split(',') if x.strip()]
        if not t:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"Hopping array error: {e}")
        t = [-0.1]  # Default on error
    try:
        t_so = [complex(x.strip()) for x in t_so_input.split(',') if x.strip()]
        if not t_so:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"Spin-orbit error: {e}")
        t_so = [0.0]  # Default on error
    ham_params = {"t": t, "t_so": t_so}
elif ham_type == "interpolation":
    st.sidebar.text("Interpolation: t(r) = a * exp(-r / b)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a_param = st.number_input("a (eV)", value=1.0, step=0.1)
    with col2:
        b_param = st.number_input("b (nm)", value=0.5, step=0.1)
    ham_params = {"a_param": a_param, "b_param": b_param}

# Button to plot Hofstadter butterfly
if st.sidebar.button("Plot Hofstadter", disabled=not st.session_state.lattice_created):
    try:
        with st.spinner("Computing Hofstadter butterfly..."):
            latt = st.session_state.latt
            if hasattr(latt, 'set_eigvals'):
                del latt.set_eigvals
                del latt.set_eigvecs
                gc.collect()
            st.session_state.hofstadter_fig = latt.plot_hofstadter(b_max, b_steps, g_factor, ham_type, **ham_params)
            st.session_state.b_max = b_max
            st.session_state.b_steps = b_steps
            st.session_state.b_values = np.linspace(0, b_max, b_steps)
        st.session_state.hofstadter_plotted = True
        st.session_state.combined_fig = None
        st.session_state.pop('dos_data', None)  # Clear previous DOS data
        st.write(f"Hofstadter: B_max={b_max} T, {b_steps} steps, g={g_factor}.")
    except Exception as e:
        st.error(f"Error plotting Hofstadter: {e}")
        logging.error(f"Error plotting Hofstadter: {e}", exc_info=True)

# Sidebar: DOS Parameters (unlocked after Hofstadter plot)
if st.session_state.hofstadter_plotted:
    st.sidebar.header("DOS Parameters")
    b_value_input = st.sidebar.text_input("Magnetic Field (T)", value="0, 20.5", help="e.g., 0, 10 will calculate DOS at 0 and 10 T")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        e_step = st.number_input("Energy steps", min_value=100, value=1000, step=100, help="Energy steps to plot DOS")  # DOS resolution
    with col2:
        smear = st.number_input("Smearing (eV)", min_value=0.0001, value=0.001, step=0.0001, format="%.4f", help="Numerial smearing to plot DOS")  
    try:
        b_value = [float(x.strip()) for x in b_value_input.split(',') if x.strip()]
        if not b_value:
            raise ValueError("No valid values")
    except ValueError as e:
        st.sidebar.error(f"Magnetic field values array error: {e}")
        b_value = [0.0]  # Default on error

    # Button to plot DOS
    if st.sidebar.button("Plot DOS"):
        try:
            with st.spinner("Computing DOS..."):
                latt = st.session_state.latt
                e_min = np.min(latt.set_eigvals) + 0.1 * np.min(latt.set_eigvals)
                e_max = np.max(latt.set_eigvals) + 0.1 * np.max(latt.set_eigvals)
                dos_fig = latt.plot_dos(b_value, e_min, e_max, e_step, smear)
                # Extract DOS data
                dos_data = []
                dos_axes = dos_fig.get_axes()[0]
                dos_lines = dos_axes.get_lines()
                for i, line in enumerate(dos_lines):
                    energy = line.get_xdata()
                    dos = line.get_ydata()
                    dos_data.append((b_value[i], energy, dos))
                st.session_state.dos_data = dos_data
                # Create combined figure
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
            logging.error(f"DOS error: {e}", exc_info=True)

# Sidebar: Spatial Map Parameters (unlocked after Hofstadter plot)
if st.session_state.hofstadter_plotted:
    st.sidebar.header("Spatial Map Parameters")
    b_value_map = st.sidebar.number_input("Magnetic Field (T)", min_value=0.0, max_value=b_max, value=0.0, step=0.1, help="Magnetic field value for eigenvector sets")
    num_eigvecs_input = st.sidebar.text_input("Eigenvectors Indices", value="0,1", help="e.g., 0,1 means |W(0|^2 + |W(1|^2)")
    mapRes = st.sidebar.slider("Resolution of spacial map", 50, 200, 100)
    smear_map = st.sidebar.number_input("Gaussian smearing (nm)", min_value=0.1, value=10.0, step=1.0, help="Gaussian basis function  exp(-|r - R_i|^2 / smear)")
    try:
        num_eigvecs = [int(x.strip()) for x in num_eigvecs_input.split(',') if x.strip()]
        if not num_eigvecs:
            raise ValueError("No valid indices")
    except ValueError as e:
        st.sidebar.error(f"Eigenvectors Indices error: {e}")
        num_eigvecs = [0]  # Default on error

    # Button to plot spatial map
    if st.sidebar.button("Plot Spatial Map"):
        try:
            with st.spinner("Computing spatial map..."):
                latt = st.session_state.latt
                st.session_state.spatial_fig = latt.plot_map(b_value_map, num_eigvecs, mapRes, smear_map)
                st.write(f"Map: B={b_value_map} T, eigvecs {num_eigvecs}.")
        except ValueError as e:
            st.error(f"Map error: {e}")
            logging.error(f"Map error: {e}", exc_info=True)

col1, col2 = st.columns(2)
bottom_container = st.container()

with col1:
    if st.session_state.lattice_fig is not None:
        buf = BytesIO()
        st.session_state.lattice_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.image(buf, caption="Lattice Plot", use_container_width=True)
        buf.seek(0)
        st.download_button(label="save plot", data=buf, file_name="lattice_plot.png", mime="image/png")

with col2:
    if st.session_state.spatial_fig is not None:
        buf = BytesIO()
        st.session_state.spatial_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.image(buf, caption="Spatial Map", use_container_width=True)
        buf.seek(0)
        st.download_button(label="save plot", data=buf, file_name="spatial_map.png", mime="image/png")

with bottom_container:
    if st.session_state.combined_fig is not None:
        buf = BytesIO()
        st.session_state.combined_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.image(buf, caption="Hofstadter Butterfly with DOS", use_container_width=True)
        buf.seek(0)
        st.download_button(label="save plot", data=buf, file_name="hofstadter_dos_plot.png", mime="image/png")
    elif st.session_state.hofstadter_fig is not None:
        buf = BytesIO()
        st.session_state.hofstadter_fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.image(buf, caption="Hofstadter Butterfly", use_container_width=True)
        buf.seek(0)
        st.download_button(label="save plot", data=buf, file_name="hofstadter_plot.png", mime="image/png")
    # Data download button (available after Hofstadter plot)
    if st.session_state.hofstadter_plotted:
        data_text = generate_data_text(
            st.session_state.latt,
            st.session_state.b_values,
            st.session_state.get('dos_data', None)
        )
        st.download_button(
            label="save data",
            data=data_text,
            file_name="hofstadter_data.txt",
            mime="text/plain"
        )

if not st.session_state.lattice_created:
    st.sidebar.write("Select lattice type and create it.")
elif not st.session_state.hofstadter_plotted:
    st.sidebar.write("Plot Hofstadter to unlock DOS and map.")

st.markdown(
    """
    <div style='text-align: center; position: fixed; bottom: 10px; width: 100%;'>
        <a href='https://github.com/danis-b/HB_lattice' target='_blank'>GitHub (danis-b)</a>
    </div>
    """,
    unsafe_allow_html=True
)
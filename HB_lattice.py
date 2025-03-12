import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class HB_lattice:

    def __init__(self, parallel=False, savefigs=False):
        self.coords = None
        self.bond_len = None
        self.parallel = parallel
        self.savefigs = savefigs

    def create_lattice(self, latt_type: str, num_cells: int, bond_len: float):
        """
        Construct a lattice with given geometry:

        Parameters:
        latt_type (str) symmetry of lattice
        num_cells (int) number of cells
        bond_len (float) bond length (in nm)
        """
        self.bond_len = bond_len
        if latt_type == "triangular":
            self.coords = self._create_triangular_lattice(num_cells, bond_len)
        elif latt_type == "square":
            self.coords = self._create_square_lattice(num_cells, bond_len)
        elif latt_type == "honeycomb":
            self.coords = self._create_honeycomb_lattice(num_cells, bond_len)
        elif latt_type == "kagome":
            print("For test only! Need to extend the idea!")
            self.coords = self._create_kagome_lattice(num_cells, bond_len)
        else:
            raise ValueError(
                "Unsupported lattice type! Available lattices are: triangular, square, honeycomb and kagome"
            )

        print(
            f"{num_cells}x{num_cells} {latt_type} lattice with {self.coords.shape[0]} sites and {bond_len} nm bond length"
        )
        self._plot_lattice()  # plot lattice immediately after creation

    def _create_triangular_lattice(self, num, bond_len):
        coords = []
        num += 1  # number of sites along one edge
        for row in range(num):
            y = row * (bond_len * np.sqrt(3) / 2)
            x_offset = row * (bond_len / 2)
            for col in range(num - row):
                x = x_offset + col * bond_len
                coords.append([x, y])
        return np.array(coords)

    def _create_square_lattice(self, num, bond_len):
        coords = []
        num += 1  # in one unit cell there are 4 atoms
        for i in range(1, num + 1):
            for j in range(1, num + 1):
                coords.append([i * bond_len, j * bond_len])
        return np.array(coords)

    def _create_honeycomb_lattice(self, num, bond_len):
        coords = []
        a1 = np.array([3 / 2 * bond_len, np.sqrt(3) / 2 * bond_len])
        a2 = np.array([3 / 2 * bond_len, -np.sqrt(3) / 2 * bond_len])
        for i in range(num + 1):
            for j in range(num + 1):
                pos1 = i * a1 + j * a2
                coords.append(pos1)
                pos2 = pos1 + np.array([bond_len, 0])
                coords.append(pos2)
        coords.pop(-1)
        coords.pop(0)
        return np.array(coords)

    def _create_kagome_lattice(self, num, bond_len):
        coords = []
        a1 = np.array([bond_len, 0])
        a2 = np.array([bond_len / 2, bond_len * np.sqrt(3) / 2])

        basis = [
            np.array([0, 0]),
            np.array([bond_len / 2, 0]),
            np.array([bond_len / 4, bond_len * np.sqrt(3) / 4]),
        ]

        num += 1  # number of unit cells along one edge
        for row in range(num):
            for col in range(num - row):
                origin = row * a2 + col * a1
                for b in basis:
                    coords.append(origin + b)

        return np.array(coords)

    def create_custom_lattice(self, file_path: str):
        _, ext = os.path.splitext(file_path)

        if ext.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif ext.lower() == ".txt" or ext.lower() == ".dat":
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        else:
            raise ValueError("Unsupported file format. Please use a .csv or .txt file.")

        if df.shape[1] != 2:
            raise ValueError(
                f"Error: file should contain two (x, y) columns (in nm), but found {df.shape[1]}."
            )
        self.coords = np.array(df.iloc[:, :])

        num_sites = self.coords.shape[0]

        unique_distances = []
        tol = 1e-4  # tolerance for comparing distances
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                r = np.linalg.norm(self.coords[i] - self.coords[j])
                if r > 0 and not any(
                    np.isclose(r, d, atol=tol) for d in unique_distances
                ):
                    unique_distances.append(r)
        unique_distances.sort()

        print(f"Custom lattice from {file_path}")
        print(f"Mind that coordinates in {file_path} must be in nm!")
        print(f"Number of sites: {num_sites}")
        print(f"3 first neighbor distances are {unique_distances[:4]} nm")
        self._plot_lattice()

    def _plot_lattice(self):
        if self.coords is None:
            raise ValueError(
                "No lattice coordinates to plot. Please create a lattice first."
            )

        num_sites = self.coords.shape[0]
        # Determine grid limits from self.coords.
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        margin_x = 0.1 * (x_max - x_min)
        margin_y = 0.1 * (y_max - y_min)
        
        if margin_y == 0: # for 1D chain along x
            margin_y = 3 * margin_x
            
        if margin_x == 0: # for 1D chain along y
            margin_x = 3 * margin_y
            
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
                    
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(
            self.coords[:, 0], self.coords[:, 1], color="blue", s=400 / num_sites
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")

        if self.savefigs:
            fig.savefig(
                "Lattice.png",
                dpi=300,
                facecolor="w",
                transparent=False,
                bbox_inches="tight",
            )

    # To DO (use sparse)
    # from scipy.sparse import csr_matrix, csc_matrix
    # from scipy.sparse.linalg import eigsh
    # def _eigensolver(self, ham_matrix):
    #     non_zero = np.count_nonzero(ham_matrix)
    #     total_elements = ham_matrix.size
    #     sparsity = non_zero / total_elements

    #     if sparsity < 0.1:
    #         print('use sparse')
    #         # Convert to CSC for better performance
    #         H_sparse = csc_matrix(ham_matrix)
    #         k = min(ham_matrix.shape[0] - 2, ham_matrix.shape[0] // 2)
    #         return eigsh(H_sparse, k=k, which='LA', return_eigenvectors=True)
    #     else:
    #         return np.linalg.eigh(ham_matrix)

    def _calc_eigenvalues(
        self, ham_type: str = "hopping", b_field: float = 0, **kwargs
    ):
        """
        Create the tight-binding Hamiltonian for the lattice using one of two methods.

        Parameters:
        ham_type (str): Which method to use. Accepts "hopping" (default) or "interpolation" .
        b_field (float): Magnetic field (in Tesla).
        **kwargs: Additional parameters for the chosen routine.
            For "hopping":
                t (list): Hopping amplitudes for different neighbor distances (in eV). Default is [1.0].
                t_so (list): Spin–orbit coupling amplitudes (in eV). Default is [0].
            For "interpolation":
                a_param (float): Prefactor for the exponential hopping term (in eV).
                b_param (float): Length scale for the exponential decay (in nm).

        Returns:
            (eigvals, eigvecs): Eigenvalues and eigenvectors of the full Hamiltonian.

        Notes:
            - This function caches geometry-dependent data in the attributes
            _bond_data_hopping or _bond_data_interpolation so that when called in a loop
            (e.g. varying b_field) the expensive computations over bonds are not repeated.
            - The Peierls phase is computed as:
              phase = exp{ -2π i * 0.242e-3 * b_field * ((x_i+x_j)(y_i-y_j)/2) }
        """
        if self.coords is None:
            raise ValueError("No coordinates found. Please create a lattice first.")

        num_sites = self.coords.shape[0]
        tol = 1e-4
        # b_field enters only through the Peierls phase and Zeeman splitting.
        phase_prefactor = -2 * np.pi * 1j * 0.242e-3 * b_field

        # Initialize Hamiltonian blocks.
        H_up = np.zeros((num_sites, num_sites), dtype=complex)
        H_dn = np.zeros((num_sites, num_sites), dtype=complex)
        H_soc = np.zeros((num_sites, num_sites), dtype=complex)

        if ham_type.lower() == "hopping":
            # Cache geometry-dependent data for the hopping method.
            if not hasattr(self, "_bond_data_hopping"):
                unique_distances = []
                bond_data = []
                # First, compute all unique nonzero distances.
                for i in range(num_sites):
                    for j in range(i + 1, num_sites):
                        r = np.linalg.norm(self.coords[i] - self.coords[j])
                        if r > 0 and not any(
                            np.isclose(r, d, atol=tol) for d in unique_distances
                        ):
                            unique_distances.append(r)
                unique_distances.sort()
                # Now, for each bond compute its neighbor index and a geometry factor used in the phase.
                for i in range(num_sites):
                    for j in range(i + 1, num_sites):
                        r = np.linalg.norm(self.coords[i] - self.coords[j])
                        neighbor_index = None
                        for idx, d in enumerate(unique_distances):
                            if np.isclose(r, d, atol=tol):
                                neighbor_index = idx
                                break
                        if neighbor_index is not None:
                            # The geometry-dependent factor inside the exponential.
                            geom_factor = (
                                (self.coords[i][0] + self.coords[j][0])
                                * (self.coords[i][1] - self.coords[j][1])
                                / 2
                            )
                            bond_data.append((i, j, neighbor_index, geom_factor, r))
                self._bond_data_hopping = (unique_distances, bond_data)
            else:
                unique_distances, bond_data = self._bond_data_hopping

            # Retrieve hopping parameters.
            t = kwargs.get("t", [0.1])
            t_so = kwargs.get("t_so", [0])
            num_hoppings = max(len(t), len(t_so))
            if num_hoppings > num_sites:
                raise ValueError(
                    "Input hopping arrays (t or t_so) are longer than the number of sites."
                )

            # Loop over precomputed bond_data.
            for i, j, neighbor_index, geom_factor, r in bond_data:
                phase = np.exp(phase_prefactor * geom_factor)
                # Hopping term.
                if neighbor_index < len(t):
                    H_up[i, j] = t[neighbor_index] * phase
                    H_up[j, i] = np.conjugate(H_up[i, j])
                    H_dn[i, j] = t[neighbor_index] * phase
                    H_dn[j, i] = np.conjugate(H_dn[i, j])
                # Spin–orbit coupling term.
                if neighbor_index < len(t_so):
                    H_soc[i, j] = t_so[neighbor_index]
                    H_soc[j, i] = t_so[neighbor_index]

        elif ham_type.lower() in ("interpolation"):
            # Cache geometry-dependent bond data for interpolation.
            if not hasattr(self, "_bond_data_interpolation"):
                bond_data = []
                for i in range(num_sites):
                    for j in range(i + 1, num_sites):
                        r = np.linalg.norm(self.coords[i] - self.coords[j])
                        geom_factor = (
                            (self.coords[i][0] + self.coords[j][0])
                            * (self.coords[i][1] - self.coords[j][1])
                            / 2
                        )
                        bond_data.append((i, j, geom_factor, r))
                self._bond_data_interpolation = bond_data
            else:
                bond_data = self._bond_data_interpolation

            # Retrieve interpolation parameters.
            a_param = kwargs.get("a_param")
            b_param = kwargs.get("b_param")
            if a_param is None or b_param is None:
                raise ValueError(
                    "For 'interpolation', a_param and b_param must be provided."
                )

            for i, j, geom_factor, r in bond_data:
                phase = np.exp(phase_prefactor * geom_factor)
                t_val = a_param * np.exp(-r / b_param)
                H_up[i, j] = t_val * phase
                H_up[j, i] = np.conjugate(H_up[i, j])
                H_dn[i, j] = t_val * phase
                H_dn[j, i] = np.conjugate(H_dn[i, j])
            # Note: H_soc remains zero in the interpolation method.

        else:
            raise ValueError("Invalid ham_type. Must be 'hopping' or 'interpolation'.")

        # Add Zeeman splitting on-site if enabled.
        zeeman_up = -5.588e-5 * self.g_factor * 0.5 * b_field
        zeeman_dn = 5.588e-5 * self.g_factor * 0.5 * b_field
        for i in range(num_sites):
            H_up[i, i] += zeeman_up
            H_dn[i, i] += zeeman_dn

        # Assemble full Hamiltonian matrix.
        H_full = np.block([[H_up, H_soc], [np.conjugate(H_soc), H_dn]])

        return np.linalg.eigh(H_full)

    def plot_hofstadter(
        self, b_max: float, b_steps: int, g_factor: float, ham_type: str, **params
    ):
        """
        Plot the Hofstadter butterfly

        Parameters:
        b_max (float): Maximum magnetic field (in Tesla).
        b_steps (int): Number of points in the magnetic field range.
        g_factor (float): g-factor used for Zeeman splitting.
        ham_type (str): Which Hamiltonian construction method to use.
                        Accepts:
                          - "hopping"  for the hopping-based method.
                          - "interpolation"  for the interpolated method.
        **params: Additional parameters for the chosen routine.
                  For "hopping": t, t_so
                  For "interpolation": a_param, b_param

        The function computes the eigenvalues for each magnetic field value
        and plots each energy level versus magnetic field.
        """
        if self.parallel:
            from joblib import Parallel, delayed

        # Initialize storage for eigenvalues and eigenvectors.
        self.set_eigvals = []
        self.set_eigvecs = []
        self.g_factor = g_factor

        print("Hofstadter plot: using Hamiltonian method:", ham_type)
        print("Loaded parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Create an array of magnetic field values.
        self.b_values = np.linspace(0, b_max, b_steps)

        # Loop over magnetic field values.
        if self.parallel:
            results = Parallel(n_jobs=-1)(
                delayed(self._calc_eigenvalues)(ham_type, b_field=b, **params)
                for b in self.b_values
            )
            self.set_eigvals = [w for (w, v) in results]
            self.set_eigvecs = [v for (w, v) in results]
        else:
            for b in self.b_values:
                w, v = self._calc_eigenvalues(ham_type, b_field=b, **params)
                self.set_eigvals.append(w)
                self.set_eigvecs.append(v)

        # Convert the list of eigenvalue arrays into a NumPy array.
        eigenvals_array = np.array(self.set_eigvals)  # shape: (b_steps, num_levels)

        # Plot each energy level versus the magnetic field.
        fig = plt.figure(figsize=(6, 4))
        for level in range(eigenvals_array.shape[1]):
            plt.plot(
                self.b_values, eigenvals_array[:, level], color="blue", linewidth=0.5
            )

        plt.xlim(self.b_values[0], self.b_values[-1])
        plt.xlabel("Magnetic field (T)")
        plt.ylabel("Energy (eV)")
        plt.title("Eigenvalues vs magnetic field")
        if self.savefigs:
            fig.savefig(
                "Eigenvalues.png",
                dpi=300,
                facecolor="w",
                transparent=False,
                bbox_inches="tight",
            )

    def _get_b_indices(self, targets, atol=None, rtol=1e-5):
        step = self.b_values[1] - self.b_values[0] if len(self.b_values) > 1 else 1
        if atol is None:
            atol = step * 0.5
        indices = []
        for t in targets:
            if t < self.b_values[0] or t > self.b_values[-1]:
                raise ValueError(
                    f"Target {t} is out of range [{self.b_values[0]}, {self.b_values[-1]}]."
                )
            idx = int(round(t / step))
            if not np.isclose(self.b_values[idx], t, rtol=rtol, atol=atol):
                diff = abs(self.b_values[idx] - t)
                raise ValueError(
                    f"No element close enough to {t} (diff: {diff} > atol {atol} with rtol {rtol})."
                )
            indices.append(idx)
        return indices

    def plot_dos(
        self,
        b_value: list = [0],  # in Tesla
        e_min: float = -5,  # eV
        e_max: float = 5,  # eV
        e_step: int = 1000,
        smear: float = 0.0001,
    ):
        if not hasattr(self, "set_eigvals"):
            raise ValueError("No eigenvalues found. Please run plot_hofstadter first!")

        num_plots = len(b_value)
        if num_plots >= len(self.set_eigvals):
            raise ValueError("Input array (b_value) is longer the number of b_steps.")

        print(f"DOS for eigenvalue sets of magnetic field : {b_value}")
        print(f"energy range from {e_min} to {e_max}")
        print(f"numerical smearing is {smear}")

        def dirac_delta(energy, kT):
            delta = np.zeros_like(energy)
            condition = np.abs(energy.real / kT) < 20
            delta[condition] = (np.exp(energy.real[condition] / kT) / kT) / (
                1 + np.exp(energy.real[condition] / kT)
            ) ** 2
            return delta

        energy_range = np.linspace(e_min, e_max, e_step)
        num_energies = len(energy_range)
        num_plots = len(b_value)
        dos = np.zeros((num_plots, num_energies))
        # find indices for b_value from self.b_values
        b_indices = self._get_b_indices(b_value)

        for plot_index, eigval_index in enumerate(b_indices):
            eigenvalues = self.set_eigvals[eigval_index]
            for j in range(num_energies):
                dos[plot_index, j] += np.sum(
                    dirac_delta(energy_range[j] - eigenvalues, smear)
                )

        fig = plt.figure(figsize=(6, 4))

        for plot_index in range(num_plots):
            plt.plot(
                energy_range,
                dos[plot_index],
                linewidth=1,
                label=f"Eigvals for {b_value[plot_index]} T",
            )  # Plot each DOS

        plt.xlim(energy_range[0], energy_range[-1])
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.legend(loc="upper right")
        if self.savefigs:
            fig.savefig(
                "DOS.png",
                dpi=300,
                facecolor="w",
                transparent=False,
                bbox_inches="tight",
            )

    def plot_map(
        self,
        b_value: float = 0,
        num_eigvecs: list = [0],
        mapRes: int = 100,
        smear: float = 10,
    ):
        """
        Plot a spatial map of the eigenstate probability density.

        Parameters:
            b_value (float): Magnetic field value for eigenvector set
            num_eigvecs (list): Indices of eigenstates to include in the map.
            mapRes (int): Resolution of the map grid.
            smear (float): Smearing parameter for the Gaussian function (default 10 nm).
        """
        # Print plotting settings once.
        print("Map plotting settings:")
        print("  Eigenstates included:", num_eigvecs)
        print("  Map resolution =", mapRes)
        print("  Gaussian smearing =", smear)
        print("  Using eigenvector set for magnetif field value:", b_value, "T")

        # find indices for b_value from self.b_values
        b_index = self._get_b_indices([b_value])[0]
        print(b_index)

        # Check that eigenvectors are available.
        if not hasattr(self, "set_eigvecs") or not self.set_eigvecs:
            raise ValueError("No eigenvectors found. Please run plot_hofstadter first!")
        try:
            eigvecs = self.set_eigvecs[b_index]
        except IndexError:
            raise ValueError(
                f"Invalid b_value {b_value}. Available values: 0 to {max(self.b_values)}"
            )

        num_sites = self.coords.shape[0]

        # Determine grid limits from self.coords.
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        margin_x = 0.1 * (x_max - x_min)
        margin_y = 0.1 * (y_max - y_min)
        
        if margin_y == 0: # for 1D chain along x
            margin_y = 3 * margin_x
            
        if margin_x == 0: # for 1D chain along y
            margin_x = 3 * margin_y
            
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y

        # Create grid.
        x = np.linspace(x_min, x_max, mapRes)
        y = np.linspace(y_min, y_max, mapRes)
        xGrid, yGrid = np.meshgrid(x, y)

        # For vectorization: reshape grid arrays to (mapRes, mapRes, 1).
        X = xGrid[..., np.newaxis]
        Y = yGrid[..., np.newaxis]
        # Extract site positions (shape: (N,)).
        site_x = self.coords[:, 0]
        site_y = self.coords[:, 1]
        # Compute the Gaussian basis functions for all sites at once.
        # Resulting phi_grid shape: (mapRes, mapRes, N)
        phi_grid = np.exp(-(((X - site_x) ** 2 + (Y - site_y) ** 2) / smear))

        # Initialize probability density map.
        z = np.zeros((mapRes, mapRes))

        for i in num_eigvecs:
            # For spinful systems, split into spin-up and spin-down parts.
            c_up = eigvecs[:num_sites, i]
            c_dn = eigvecs[num_sites:, i]
            psiR_up = np.sum(phi_grid * c_up, axis=2)
            psiR_dn = np.sum(phi_grid * c_dn, axis=2)
            z += np.abs(psiR_up) ** 2 + np.abs(psiR_dn) ** 2
            print(f"  State {i}: Eigenvalue = {self.set_eigvals[b_index][i]}")

        # Plot the probability density map.
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.pcolormesh(x, y, z, cmap="Reds", shading="nearest")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")

        if self.savefigs:
            fig.savefig(
                "Eigenvectors_map.png",
                dpi=300,
                facecolor="w",
                transparent=False,
                bbox_inches="tight",
            )
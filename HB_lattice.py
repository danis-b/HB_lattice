import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class HB_lattice:

    def __init__(self, g_factor=2):
        self.coords = None
        self.bond_len = None
        self.evals = None
        self.g_factor = g_factor

    def create_lattice(self, type: str, num_cells: int, bond_len: float):
        """
        Construct a lattice with given geometry:
        
        Parameters:
        type (str) symmetry of lattice
        num_cells (int) number of cells or sites (depending in latttice)
        bond_len (float) bond length (in nm)
        """
        self.bond_len = bond_len
        if type == "triangular":
            self.coords = self._create_triangular_lattice(num_cells, bond_len)
        elif type == "square":
            self.coords = self._create_square_lattice(num_cells, bond_len)
        elif type == "honeycomb":
            print("honeycomb", num_cells, bond_len)
            self.coords = self._create_honeycomb_lattice(num_cells, bond_len)
        elif type == "kagome":
            self.coords = self._create_kagome_lattice(num_cells, bond_len)
        else:
            raise ValueError("Unsupported lattice type")
        
        print(f"{num_cells}x{num_cells} {type} lattice with {bond_len} bond length was constructed")

    def _create_triangular_lattice(self, num, bond_len):
        coords = []
        half_num = num // 2
        for i in range(-half_num, half_num + 1):
            for j in range(-half_num, half_num + 1):
                x = i * bond_len
                y = j * bond_len * np.sqrt(3) / 2
                if j % 2 == 1:
                    x += bond_len / 2
                coords.append([x, y])
        return np.array(coords)

    def _create_square_lattice(self, num, bond_len):
        coords = []
        half_num = num // 2
        for i in range(-half_num, half_num + 1):
            for j in range(-half_num, half_num + 1):
                coords.append([i * bond_len, j * bond_len])
        return np.array(coords)

    def _create_honeycomb_lattice(self, num, bond_len):
        coords = []
        tau1 = np.array([3 / 2 * bond_len, np.sqrt(3) / 2 * bond_len])
        tau2 = np.array([3 / 2 * bond_len, -np.sqrt(3) / 2 * bond_len])
        for i in range(num + 1):
            for j in range(num + 1):
                pos1 = i * tau1 + j * tau2
                coords.append(pos1)
                pos2 = pos1 + np.array([bond_len, 0])
                coords.append(pos2)
        coords.pop(-1)
        coords.pop(0)
        return np.array(coords)

    def _create_kagome_lattice(self, num, bond_len):
        coords = []
        tau1 = np.array([bond_len, 0])
        tau2 = np.array([bond_len / 2, bond_len * np.sqrt(3) / 2])
        for i in range(num + 1):
            for j in range(num + 1):
                origin = i * tau1 + j * tau2
                pos1 = origin + np.array([0, 0])
                pos2 = origin + np.array([bond_len / 2, 0])
                pos3 = origin + np.array([bond_len / 4, bond_len * np.sqrt(3) / 4])
                coords.append(pos1)
                coords.append(pos2)
                coords.append(pos3)

        return np.array(coords)

    def create_custom_lattice(self, file_path: str):
        df = pd.read_csv(file_path)
        if df.shape[1] != 2:
            raise ValueError(
                f"Error: csv file should contain two (x,y) columns, but found {df.shape[1]}."
            )
        self.coords = np.array(df.iloc[:, :])
        
        print(f"Custom lattice from {file_path} was constructed")
        print(f"Number of sites: {self.coords.shape[0]}")

    def plot_lattice(self):
        if self.coords is None:
            raise ValueError(
                "No lattice coordinates to plot. Please create a lattice first."
            )

        plt.figure(figsize=(4, 4))
        plt.scatter(self.coords[:, 0], self.coords[:, 1], color="blue", s=100)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X (A)")
        plt.ylabel("Y (A)")
        plt.title("Lattice")
        plt.show()

    def hamiltonian_via_hopping(self, t: list, t_so = [0], b_field = 0):
        """
        Construct a spinful Hamiltonian from hopping, spin–orbit, and magnetic field terms.
        
        Parameters:
        t (list): Array of hopping amplitudes for different neighbor distances (in eV).
        t_so (list): Array of spin–orbit coupling hoppings for different neighbor distances (in eV).
        b_field (float): Magnetic field (in Tesla).

        The magnetic field enters in two ways:
        1. A Peierls phase is attached to the hopping terms:
            phase = exp{ -2π i (0.242e-3 * b_field) * ((x_i+x_j)(y_i-y_j)/2) }.
        2. Zeeman splitting adds an on–site shift:
            for spin up:   −5.588e-5 * self.g_factor * 0.5 * b_field,
            for spin down: +5.588e-5 * self.g_factor * 0.5 * b_field.
        """
        if self.coords is None:
            raise ValueError(
                "No coordinates are found. Please create a lattice  first."
            )
        
        num_sites = self.coords.shape[0]
        num_hoppings = max(len(t), len(t_so))

        if num_hoppings > num_sites:
            raise ValueError(
                "Input hopping arrays (t or t_so) is longer than the number of sites in the lattice."
            )

        b = 0.242e-3 * b_field

        # Initialize matrices for the two spin sectors and for the spin–orbit (SOC) term.
        H_up = np.zeros((num_sites, num_sites), dtype=complex)
        H_dn = np.zeros((num_sites, num_sites), dtype=complex)
        H_soc = np.zeros((num_sites, num_sites), dtype=complex)

        # Loop over pairs of sites (using i<j, then fill in the Hermitian conjugate).
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                # Determine which neighbor (in order) this bond represents.
                neighbor_index = j - i - 1

                # Compute the Peierls phase
                phase = np.exp(-2 * np.pi * 1j * b * 
                    ((self.coords[i][0] + self.coords[j][0]) * (self.coords[i][1] - self.coords[j][1]) / 2))

                # Add the standard hopping (for both spin-up and spin-down sectors)
                if neighbor_index < len(t):
                    hopping = t[neighbor_index]
                    H_up[i, j] = hopping * phase
                    H_up[j, i] = np.conjugate(H_up[i, j])
                    H_dn[i, j] = hopping * phase
                    H_dn[j, i] = np.conjugate(H_dn[i, j])

                # Add the SOC term to the off-diagonal (spin-flip) block.
                if neighbor_index < len(t_so):
                    soc = t_so[neighbor_index]
                    H_soc[i, j] = soc
                    

        # Add the Zeeman splitting on-site:
        for i in range(num_sites):
            H_up[i, i] += -5.588e-5 * self.g_factor * 0.5 * b_field
            H_dn[i, i] += 5.588e-5 * self.g_factor * 0.5 * b_field

        # Construct the full spinful Hamiltonian as a block matrix:
        H_full = np.block([[H_up, H_soc], [H_soc.conj().T, H_dn]])

        # Diagonalize the full Hamiltonian.
        self.evals = np.linalg.eigvalsh(H_full)
        
        

    def hamiltonian_via_interpolation(self, a: float, b: float):
        print(f"Hamiltonian will be constructd using f(ij) = {a} exp(-{b}d_ij])")
        num_sites = self.coords.shape[0]
        self.ham_matrix = np.zeros((num_sites, num_sites), dtype=complex)
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                self.ham_matrix[i, j] = a * np.exp(-b * dist)
                self.ham_matrix[j, i] = a * np.exp(-b * dist)


    def plot_dos(self, emin: float, emax: float, smear: float):
        if self.evals is None:
            raise ValueError(
                "No eigenvalues are found. Please calculate eigenvalues first."
            )

        def dirac_delta(energy, kT):
            if np.abs(energy.real / kT) < 20:
                delta = (np.exp(energy.real / kT) / kT) / ( 1 + np.exp(energy.real / kT)) ** 2
            else:
                delta = 0
            return delta

        eigvals = self.evals
        energy_range = np.linspace(emin, emax, 1000)
        dos = np.zeros_like(energy_range)

        for i in range(1000):
            for j in range(len(eigvals)):
                dos[i] += dirac_delta(energy_range[i] - eigvals[j], smear)

        plt.plot(energy_range, dos)
        plt.xlabel("Energy")
        plt.ylabel("Density of States")
        plt.title("Density of States")
        plt.show()

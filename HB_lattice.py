import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class HB_lattice:

    def __init__(self):
        self.coords = None
        self.bond_len = None
        self.ham_matrix = None

    def create_lattice(self, type: str, num_cells: int, bond_len: float):
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

    def hamiltonian_hopping(self, t: list):
        print(f"Hamiltonian will be constructd with hopping: {t}")
        num_elements = len(t)
        num_sites = self.coords.shape[0]

        if num_elements > num_sites:
            raise ValueError(
                "Input array t is longer than the number of sites in the lattice."
            )

        self.ham_matrix = np.zeros((num_sites, num_sites), dtype=complex)
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                if j - i - 1 < num_elements:
                    self.ham_matrix[i, j] = t[j - i - 1]
                    self.ham_matrix[j, i] = np.conj(t[j - i - 1])

    def hamiltonian_interpolate(self, a: float, b: float):
        print(f"Hamiltonian will be constructd using f(ij) = {a} exp(-{b}d_ij])")
        num_sites = self.coords.shape[0]
        self.ham_matrix = np.zeros((num_sites, num_sites), dtype=complex)
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                self.ham_matrix[i, j] = a * np.exp(-b * dist)
                self.ham_matrix[j, i] = a * np.exp(-b * dist)

    def add_spinorb(self, t: list):
        if self.ham_matrix is None:
            raise ValueError(
                "No Hamiltonian matrix found. Please create a Hamiltonian first."
            )

        print(f"Hamiltonian will be constructd with hopping: {t}")
        num_elements = len(t)

        num_sites = self.coords.shape[0]
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                if j - i - 1 < num_elements:
                    self.ham_matrix[i, j] = t[j - i - 1]
                    self.ham_matrix[j, i] = np.conj(t[j - i - 1])

    def calc_eigenvalues(self, B):
        b = 0.242e-3 * B * self.a**2  # unitless  b = B * a^2 * e/h

        Ham_up = np.zeros((self.N, self.N), dtype=complex)
        Ham_ud = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                phase = np.exp(
                    -2
                    * np.pi
                    * 1j
                    * b
                    * (self.coord[i][0] + self.coord[j][0])
                    * (self.coord[i][1] - self.coord[j][1])
                    / 2
                )
                r = np.linalg.norm(self.coord[i] - self.coord[j])
                if r == 1:
                    Ham_up[i, j] = self.t_nn * phase  ## nn
                if np.abs(r - 1.41421) < 1e-4:
                    Ham_up[i, j] = self.t_nnn * phase  # nnn

        Ham_dn = np.copy(Ham_up)

        # Zeeman splitting
        for i in range(self.N):
            Ham_up[i, i] += -5.588e-5 * self.g * 0.5 * B
            Ham_dn[i, i] += 5.588e-5 * self.g * 0.5 * B

        # Non-diagonal components due to nn spin-orbit term t_soc
        for i in range(self.N):
            for j in range(self.N):
                r = np.linalg.norm(self.coord[i] - self.coord[j])
                if r == 1:
                    Ham_ud[i, j] = self.t_soc

        Ham = np.block([[Ham_up, Ham_ud], [np.conj(Ham_ud), Ham_dn]])

        evals, evects = np.linalg.eigh(Ham)

        return evals, evects

    def plot_dos(self, emin: float, emax: float, smear: float):
        if self.ham_matrix is None:
            raise ValueError(
                "No Hamiltonian matrix to plot DOS. Please create a Hamiltonian first."
            )

        def dirac_delta(energy, kT):
            if np.abs(energy.real / kT) < 20:
                delta = (np.exp(energy.real / kT) / kT) / (
                    1 + np.exp(energy.real / kT)
                ) ** 2
            else:
                delta = 0
            return delta

        eigvals = np.linalg.eigvalsh(self.ham_matrix)
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

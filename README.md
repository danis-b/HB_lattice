# Theory
Hofstadter butterfly for 2D finite lattices $n\times n$ with Zeeman term:

$$  \hat{H} = \sum_{ij, \sigma \sigma^{\prime}} t^{ \sigma \sigma^{\prime}}_{ij} \hat{a}^{\dagger \sigma }_i \hat{a}^{ \sigma^{\prime}}_j  - \frac{1}{2} g \mu_B B_z \sum \hat{a}^{\dagger \sigma }_i \sigma^z  \hat{a}^{\sigma^{\prime}}_i.  $$

Peierls substitutions is taken into account  via phase factor:

$$ t_{ij} \rightarrow  t_{ij} e^{\gamma_{ij}}, $$

$$ \gamma_{ij} = \Phi_{ij} / \Phi_0 = \frac{\int B_z dS_{ij}}{h/e} =  -2 \pi i \frac{e}{h} B_z \frac{1}{2} (x_i + x_j) (y_i - y_j).$$

# Dependencies
jupyter-notebook, numpy, matplotlib, pandas, (joblib for parallel case)

# Usage
class HB_lattice contains the following methods: 

* create_lattice(type, num_cells, bond_len) creates and plots the lattice with given properties
  * type (str): symmetry of lattice (triangular, square, honeycomb)
  *  num_cells (int): number of cells
  *  bond_len (float): bond length (in nm)
* create_custom_lattice(file_path) creates the lattice using file_path csv or txt  file.  File should contain two (x, y) columns (in nm) with sites coordinates.
* plot_hofstadter(b_max, b_steps, g_factor, ham_type, **params) calculate and plot eigenvalues varying magnetic field. 
  *  b_max (float): maximum magnetic field (in Tesla).
  *  b_steps (int): Number of points in the magnetic field range (0, b_max)
  *  g_factor (float): g-factor used for Zeeman splitting
  *  ham_type (str): type of Hamiltonian construction method to use. There are two ways: "hopping" and "interpolation" with the following parameters
  *  t (list): array of hoppings for different neighbor distances (in eV).
  *  t_so (list): array of spin–orbit coupling hoppings for different neighbor distances (in eV).
  *  a_param and b_param (float): parameter of interpolation (in eV)  t_ij = a_param * exp ( - r_ij / b_param), r_ij - distance between sites
* plot_dos(num_eigvals, e_min, e_max, e_step, smear) plot densities of states 
  * num_eigvals (list): eigenvalue sets of magnetic field to plot DOS
  * e_min (float), e_max (float), e_step (int)  energy range np.linspace(e_min, e_max, e_step) to plot DOS
  * smear (float): numerial smearing
* plot_map(num_eigvecs, mapRes, smear, b_index) plots a spatial map of the eigenstate probability density.
  * num_eigvecs (list): indices of eigenstates  to include in the map.
  * mapRes (int): resolution of the map grid.
  * smear (float): smearing parameter for the Gaussian function (default 10 nm).
  * b_index (int): Index of the eigenvector set to use (default 0)




# Examples

Square, triangular and honeycomb lattices [[1](https://pubs.aip.org/aapt/ajp/article-abstract/72/5/613/1038951/Landau-levels-molecular-orbitals-and-the?redirectedFrom=fulltext)]:
t = 0.01 eV; num_cells = 1, 15; bond_len = 10 nm;
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/Results.png)

2x2 lattice square lattice with Peierls & Zeeman terms:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/2x2_square.png)




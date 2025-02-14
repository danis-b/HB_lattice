# Theory
Hofstadter butterfly for 2D finite lattices $n\times n$ with Zeeman term:

$$  \hat{H} = \sum_{ij, \sigma \sigma^{\prime}} t^{ \sigma \sigma^{\prime}}_{ij} \hat{a}^{\dagger \sigma }_i \hat{a}^{ \sigma^{\prime}}_j  - \frac{1}{2} g \mu_B B_z \sum \hat{a}^{\dagger \sigma }_i \sigma^z  \hat{a}^{\sigma^{\prime}}_i.  $$

Peierls substitutions is taken into account  via phase factor:

$$ t_{ij} \rightarrow  t_{ij} e^{\gamma_{ij}}, $$

$$ \gamma_{ij} = \Phi_{ij} / \Phi_0 = \frac{\int B_z dS_{ij}}{h/e} =  -2 \pi i \frac{e}{h} B_z \frac{1}{2} (x_i + x_j) (y_i - y_j).$$

# Dependencies
jupyter-notebook, numpy, matplotlib, pandas

# Usage
class HB_lattice contains the following methods: 

* create_lattice(type, num_cells, bond_len) creates the lattice with given properties
  * type (str): symmetry of lattice (triangular, square, honeycomb)
  *  num_cells (int): number of cells
  *  bond_len (float): bond length (in nm)
* create_custom_lattice(file_path) creates the lattice using file_path csv-file.  File should contain two (x, y) columns (in nm) with sites coordinates.
* eigenvalues_via_hopping (t, t_so, b_field, add_peierls, add_zeeman) creates hamiltonian via hopping arrays
  * t (list): array of hoppings for different neighbor distances (in eV).
  * t_so (list): array of spin–orbit coupling hoppings for different neighbor distances (in eV).
  * b_field (float): magnetic field value (in Tesla).
  * add_peierls (bool): If True, include the Peierls phase in the hopping terms.
  * add_zeeman (bool): If True, add Zeeman splitting with given g-factor on-site. 
* eigenvalues_via_interpolation(a_param, b_param, b_field, add_peierls, add_zeeman) creates hamiltonian via interpolation function
  * a_param and b_param (float): parameter of interpolation (in eV)  t_ij = a_param * exp ( - r_ij / b_param), r_ij - distance between sites
* plot_dos(energy_range, eigvals, smear) static method to plot densities of states 
  * energy_range (list): energy range to calculate dos
  * eigvals (list): list of calculated eigenvalues
  * smear (float): numerial smearing




# Examples

Square, triangular and honeycomb lattices [[1](https://pubs.aip.org/aapt/ajp/article-abstract/72/5/613/1038951/Landau-levels-molecular-orbitals-and-the?redirectedFrom=fulltext)]:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/Results.png)

2x2 lattice square lattice with Peierls & Zeeman terms:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/2x2_square.png)




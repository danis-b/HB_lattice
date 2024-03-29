# Theory
Hofstadter butterfly for square finite lattices $n\times n$ with Zeeman term [[1](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.155412)]:

$$  \hat{H} = \sum_{ij, \sigma \sigma^{\prime}} t^{ \sigma \sigma^{\prime}}_{ij} \hat{a}^{\dagger \sigma }_i \hat{a}^{ \sigma^{\prime}}_j  - \frac{1}{2} g \mu_B B_z \sum \hat{a}^{\dagger \sigma }_i \sigma^z  \hat{a}^{\sigma^{\prime}}_i.  $$

Model takes into accoun only nnn hopping $t_{nnn}$ and nn hopping $t_{nn}^{ \sigma \sigma^{\prime}}$ with diagonal $t_{nn}$ and non-diagonal part $t_{nn}^{SOC}$ due to spin-orbit term.

Peierls substitutions is taken into account  via phase factor, where $a$ is lattice spacing of square lattice:

$$ t_{ij} \rightarrow  t_{ij} e^{\gamma_{ij}}, $$

$$ \gamma_{ij} = -2 \pi i \frac{e}{h} a^2 \frac{1}{2} (x_i + x_j) (y_i - y_j).$$

# Dependencies
jupyter-notebook, numpy, matplotlib


# Examples

2x2 lattice with nearest neighbor hopping term:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/2x2.png)

2x2 lattice with nearest neighbor hopping term (spin-orbit coupling included):
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/2x2_soc.png)

3x3 lattice with nearest neighbor hopping term:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/3x3.png)

9x9 lattice with nearest neighbor hopping term:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/9x9.png)



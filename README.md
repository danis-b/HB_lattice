# Theory
Hofstadter butterfly for 2D finite lattices $n\times n$ with Zeeman term:

$$  \hat{H} = \sum_{ij, \sigma \sigma^{\prime}} t^{ \sigma \sigma^{\prime}}_{ij} \hat{a}^{\dagger \sigma }_i \hat{a}^{ \sigma^{\prime}}_j  - \frac{1}{2} g \mu_B B_z \sum \hat{a}^{\dagger \sigma }_i \sigma^z  \hat{a}^{\sigma^{\prime}}_i.  $$

Peierls substitutions is taken into account  via phase factor:

$$ t_{ij} \rightarrow  t_{ij} e^{\gamma_{ij}}, $$

$$ \gamma_{ij} = -2 \pi i \frac{e}{h} \frac{1}{2} (x_i + x_j) (y_i - y_j).$$

# Dependencies
jupyter-notebook, numpy, matplotlib, pandas


# Examples

Square, triangular and honeycomb lattices [[1](https://pubs.aip.org/aapt/ajp/article-abstract/72/5/613/1038951/Landau-levels-molecular-orbitals-and-the?redirectedFrom=fulltext)]:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/Results.png)

2x2 lattice square lattice:
![alt text](https://github.com/danis-b/HB_lattice/blob/main/Examples/2x2_square.png)




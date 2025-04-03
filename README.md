# SparseDiskFEM.jl

This repository implements the numerical examples found in:

"A sparse hierarchical hp-finite element method on disks and annuli", Ioannis. P. A. Papadopoulos and Sheehan Olver (2024). https://arxiv.org/abs/2402.12831

We numerically approximate the solutions to variable coefficient Helmholtz and Schrödinger equations on disks and annuli. The hp-FEM is very-high-order, e.g. p=200 but the resulting matrices are sparse.

This package heavily utilises RadialPiecewisePolynomials.jl for its implementation of a sparse hp-FEM basis for disks and annuli.

|Figure(s)|File: examples/|
|:-:|:-:|
|1|[spy-plots.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/spy-plots.jl)|
|2|[3d-cylinder-v2.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/3d-cylinder-v2.jl)|
|3|[basis-slice.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/basis-slice.jl)|
|6, 7|[plane-wave.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/plane-wave.jl)|
|8, 9, 10|[high-frequency.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/high-frequency.jl)|
|11, 12, 13|[schrodinger-harmonic-oscillator.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/schrodinger-harmonic-oscillator.jl)|
|14|[non-separable.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/non-separable.jl)|
|15|[hp-refinement.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/hp-refinement.jl)|
|16, 17|[3d-cylinder.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/3d-cylinder.jl)|

## Contact
Ioannis Papadopoulos: papadopoulos@wias-berlin.de
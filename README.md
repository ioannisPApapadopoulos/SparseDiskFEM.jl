# SparseDiskFEM.jl

This repository implements the numerical examples found in:

(1) "A sparse hierarchical hp-finite element method on disks and annuli", Ioannis. P. A. Papadopoulos and Sheehan Olver (2024).

We numerically approximate the solution to equations involving fractional Laplacian operators via frame approach. This approach reduces solving an equation with nonlocal terms to an interpolation problem for the right-hand side. We find expansions via a truncated SVD which alleviates the perceived ill-conditioning.

This package heavily utilises RadialPiecewisePolynomials.jl for its implementation of a sparse hp-FEM basis for disks and annuli.

|Figure|File: examples/|
|:-:|:-:|
|1|[spy-plots.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/spy-plots.jl)|
|2|[adi-disk-hp-fem-alternate.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/adi-disk-hp-fem-alternate.jl)|
|3|[basis-slice.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/basis-slice.jl)|
|6, 7|[plane-wave.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/plane-wave.jl)|
|8, 9, 10|[high-frequency.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/high-frequency.jl)|
|11, 12, 13|[schrodinger-harmonic-oscillator.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/schrodinger-harmonic-oscillator.jl)|
|14, 15[adi-disk-hp-fem.jl](https://github.com/ioannisPApapadopoulos/SparseDiskFEM.jl/blob/main/examples/adi-disk-hp-fem.jl)|

## Contact
Ioannis Papadopoulos: papadopoulos@wias-berlin.de
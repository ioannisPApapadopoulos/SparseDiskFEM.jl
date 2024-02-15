using RadialPiecewisePolynomials
using Plots

N = 41
K = 3 # number of cells
points = range(0,1,K+1)
F = ContinuousZernike(N, points)
F0 = F.Fs[1];


M = F0' * F0;
D = Derivative(axes(F0,1))
A = (D*F0)' * (D*F0);

Ms = Matrix(M); Ms[abs.(Ms) .> 1e-12] .= 1.0; Ms[abs.(Ms) .< 1e-12] .= 0.0;
Plots.spy(Ms, markersize=3)
Plots.savefig("spy-M.pdf")

As = Matrix(A); As[abs.(As) .> 1e-12] .= 1.0; As[abs.(As) .< 1e-12] .= 0.0;
Plots.spy(As, markersize=3)
Plots.savefig("spy-A.pdf")
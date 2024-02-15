using RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials, LinearAlgebra
using PyPlot, Plots, LaTeXStrings
using SparseDiskFEM
import RadialPiecewisePolynomials: findblockindex, blockedrange, Fill, oneto

"""
Section 6.3 "Time-dependent Schrödinger equation"

⚠ Example is no longer included in Section 6.3.

Domain is Ω = {0 ≤ r ≤ 50} and we are solving the eigenvalue problem
    (-Δ + r²) u(x,y,t) = E u(x,y,t).
on the disk with zero Dirichlet bcs at r=50, 
"""

function ψ(xy)
    x, y = first(xy), last(xy)
    H = Normalized(Hermite())
    H[x,nx+1] * H[y,ny+1] * exp(-(x^2+y^2)/2)
end

points = [0; [50*1.2^(-n) for n in 15:-1:0]]; K = length(points)-1;
N=100; Φ = ContinuousZernike(N, points);

V(r²) = r² # quadratic well

@time M = Φ' * Φ; # <v, u>, v, u ∈ Φ
@time wM = Φ' * (V.(axes(Φ,1)) .* Φ); # <v, V(r²) u>, v, u ∈ Φ
D = Derivative(axes(Φ,1));
@time nΔ = (D*Φ)' * (D*Φ); # <∇v, ∇u>, v, u ∈ Φ

A = Matrix.(nΔ .+ wM);
zero_dirichlet_bcs!(Φ, A)
M = Matrix.(M)
zero_dirichlet_bcs!(Φ, [M...])

# Solve the generalized eigenvalue problem
Es = eigen.(A, M);
evs = [E.values[2:end] for E in Es];

# Eigenvalues are ordered according to degree
# Here we extract out the ordering.
tEs = 2*Vector(Vcat((Fill.(1:100, 1:100))...))
bl = [findblockindex(blockedrange(oneto(∞)), j) for j in 1:sum(1:N)]
ℓ = [bl[j].I[1]-1 for j in 1:sum(1:N)] # degree
k = [bl[j].α[1] for j in 1:sum(1:N)]   # index of degree
ms = [iseven(ℓ[j]) ? k[j]-isodd(k[j]) : k[j]-iseven(k[j]) for j in 1:sum(1:N)] # m-mode
js = isodd.(k .+ ℓ)
deg = (ℓ .- ms) .÷ 2 .+ 1

# Compute error between computed and actual eigenvalues
errs = []
for (m,j,k,i) in zip(ms, js, deg, 1:sum(1:N))
    append!(errs, [abs(evs[2m+j][k] - tEs[i])])
end

errs[errs.==0] .= NaN

first(findall(x->x>1e-10, errs ./ tEs))
first(findall(x->x>1e-5, errs ./ tEs))
first(findall(x->x>1e-1, errs ./ tEs))

tEs[1324]
tEs[2848]
tEs[4552]

Plots.plot(tEs, errs ./ tEs,
    linetype=:scatter,
    ylabel=L"$|E_h - E|/E$",
    xlabel=L"$E$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    markersize = 2,
    marker=:circle,
    legend=:none,
    ylim=[1e-17,1e0],
    yticks=[1e-15,1e-10,1e-5,1e0],
    yscale=:log10,
)
Plots.savefig("schrodinger-eigs.pdf")
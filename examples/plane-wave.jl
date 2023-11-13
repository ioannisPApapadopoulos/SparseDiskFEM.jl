using RadialPiecewisePolynomials
using PyPlot, Plots, DelimitedFiles
import ForwardDiff: derivative
using SparseDiskFEM # plotting routines
"""
This script implements the screened Poisson equation of

"Plane wave with discontinuous coefficients and data" (section 5.1).

Here we have a right-hand side and a Helmholtz coefficient that have a jump in the radial direction at r=1/2.
"""

# ρ is the inner radius on the annulus cell
# λ₀ is the jump coefficient on the inner cell
# λ₁ is the jump coefficient on the outer cell
ρ, λ₀, λ₁ = 0.5, 1e-2, 5e1
λ(r) = r ≤ ρ ? λ₀ : λ₁


ũ(r) = r ≤ ρ ? (λ₀*r^2/4 + (λ₁ - λ₀)*ρ^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(ρ)) : (λ₁*r^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(r))

# Check that Δũ(r) = λ(r)
Δũ(r) =
    (derivative(r->derivative(r->ũ(r) , r),r) 
    + derivative(r->ũ(r) , r)/r)
@assert Δũ(0.2) ≈ λ₀
@assert Δũ(0.7) ≈ λ₁

# Exact solution
k = 50
uₑ(r,θ) = sin(k*(r*cos(θ))) * ũ(r)
function uₑ_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    uₑ(r,θ)
end

# Use ForwardDiff to compute the RHS
Δuₑ(r,θ) =
    (derivative(r->derivative(r->uₑ(r,θ), r),r) 
    + derivative(r->uₑ(r,θ), r)/r 
    + derivative(θ->derivative(θ->uₑ(r,θ), θ),θ)/r^2)

rhs(r,θ) =  -Δuₑ(r,θ)/k + λ(r)*uₑ(r,θ)

# RHS in Cartesian coordinates
function rhs_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs(r,θ)
end

s = ρ^(-1/9)
points = [0; reverse([s^(-j) for j in 0:9])]
Nₕ = length(points)-1


# Ψ = FiniteZernikeBasis(1000, [0.0;1.0], 0, 0);
# fp = Ψ \ rhs_xy.(axes(Ψ,1));
# (θs, rs, vals) = finite_plotvalues(Ψ, fp)
# _, err = inf_error(Ψ, θs, rs, vals, rhs_xy);
# err

Φ = FiniteContinuousZernike(150, points);
Ψ = FiniteZernikeBasis(150, points, 0, 0);
fz = Ψ \ rhs_xy.(axes(Ψ,1));
(θs, rs, vals) = finite_plotvalues(Ψ, fz, N=300)
vals_, err = inf_error(Ψ, θs, rs, vals, rhs_xy);
err
plot(Ψ, θs, rs, vals, ttl=L"f(x,y)")
PyPlot.savefig("plane-wave-rhs.png", dpi=500)
slice_plot(60, θs, rs, vals, points, ylabel=L"$f(x,y)$")
Plots.savefig("plane-wave-rhs-slice.pdf")

Λ = piecewise_constant_assembly_matrix(Φ, λ);
D = Derivative(axes(Φ,1));
A = (D*Φ)' * (D*Φ);
G = (Φ' * Ψ);

K = Matrix.(A ./ k .+ Λ);
Mf = G .* fz
zero_dirichlet_bcs!(Φ, K)
zero_dirichlet_bcs!(Φ, Mf)

errors = []
θs, rs, vals = [], [], []
for N in 20:10:150
    Ns = getNs(N)
    
    Kn = [K[j][1:(n-1)*Nₕ, 1:(n-1)*Nₕ] for (n,j) in zip(Ns,1:lastindex(K))];
    Mfn = [Mf[j][1:(n-1)*Nₕ] for (n,j) in zip(Ns,1:lastindex(Mf))];# right-hand side

    u = Kn .\ Mfn;
    (θs, rs, vals) = finite_plotvalues(Φ, u, N=300);
    _, err = inf_error(Φ, θs, rs, vals, uₑ_xy);
    append!(errors, [err])
    writedlm("errors-plane-wave.log", errors)

    print("Computed coefficients for N=$N \n")
end

plot(Φ, θs, rs, vals, ttl=L"u(x,y)") # plot
PyPlot.savefig("plane-wave-sol.png", dpi=500)
slice_plot(60, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("plane-wave-sol-slice.pdf")


###
# Convergence plot
###
errors = readdlm("errors-plane-wave.log")

ps = 20:10:150
Plots.plot(ps, errors,
    linewidth=3,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$N_p$",
    legend=:none,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    color=3,
    gridlinewidth = 2
)
Plots.savefig("plane-wave-convergence.pdf")
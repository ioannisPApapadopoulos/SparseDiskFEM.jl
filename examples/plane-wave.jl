using RadialPiecewisePolynomials
using PyPlot, Plots, DelimitedFiles
import ForwardDiff: derivative
import RadialPiecewisePolynomials: _getMs_ms_js
using SparseDiskFEM # plotting routines
"""
This script implements the Helmholtz example of

"Discontinuous variable coefficients and data on a disk" (section 7.3).

We are solving (-Δ + λ(x,y)) u(x,y) = 
on a disk.

Here we have a right-hand side and a Helmholtz coefficient that has a jump in the radial direction at r=1/2.

"""

# ρ is the inner radius on the annulus cell
# λ₀ is the jump coefficient on the inner cell
# λ₁ is the jump coefficient on the outer cell
ρ, λ₀, λ₁ = 0.5, 1e-2, 5e1
λ(r) = r ≤ ρ ? λ₀ : λ₁

# Δũ(r) = λ(r)
ũ(r) = r ≤ ρ ? (λ₀*r^2/4 + (λ₁ - λ₀)*ρ^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(ρ)) : (λ₁*r^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(r))

# Exact solution
k = 50
ua(r,θ) = sin(k*(r*cos(θ))) * ũ(r)
function ua_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    ua(r,θ)
end

# Use ForwardDiff to compute the RHS
rhs_(r,θ) =  (
    (-derivative(r->derivative(r->ua(r,θ), r),r) 
    - derivative(r->ua(r,θ), r)/r 
    - derivative(θ->derivative(θ->ua(r,θ), θ),θ)/r^2)/k
    + λ(r)*ua(r,θ)
)

# RHS in Cartesian coordinates
function rhs_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs_(r,θ)
end

s = ρ^(-1/9)
points = [0; reverse([s^(-j) for j in 0:9])]
K = length(points)-1


# Ψ = FiniteZernikeBasis(1000, [0.0;1.0], 0, 0);
# fp = Ψ \ rhs_xy.(axes(Ψ,1));
# (θs, rs, vals) = finite_plotvalues(Ψ, fp)
# _, err = inf_error(Ψ, θs, rs, vals, rhs_xy);
# err

Φ = FiniteContinuousZernike(150, points);
Ψ = FiniteZernikeBasis(150, points, 0, 0);
fz = Ψ \ rhs_xy.(axes(Ψ,1));
(θs, rs, vals) = finite_plotvalues(Ψ, fz, N=300)
_, err = inf_error(Ψ, θs, rs, vals, rhs_xy);
plot(Ψ, θs, rs, vals, ttl=L"f(x,y)")
PyPlot.savefig("plane-wave-rhs.png", dpi=500)

slice_plot(60, θs, rs, vals, points, ylabel=L"$f(x,y)$")
Plots.savefig("plane-wave-rhs-slice.pdf")

Λ = piecewise_constant_assembly_matrix(Φ, λ);
D = Derivative(axes(Φ,1));
Δ = (D*Φ)' * (D*Φ);
G = (Φ' * Ψ);

A = Matrix.(Δ ./ k .+ Λ);
Mf = G .* fz
zero_dirichlet_bcs!(Φ, A)
zero_dirichlet_bcs!(Φ, Mf)

errors = []
θs, rs, vals = [], [], []
for N in 20:10:150
    Ms = getNs(N)
    
    An = [A[j][1:(n-1)*K, 1:(n-1)*K] for (n,j) in zip(Ms,1:lastindex(A))];
    Mfn = [Mf[j][Block.(1:n)] for (n,j) in zip(Ms,1:lastindex(Mf))];# right-hand side
    
    u = An .\ Mfn;
    (θs, rs, vals) = finite_plotvalues(Φ, u, N=300);
    _, err = inf_error(Φ, θs, rs, vals, ua_xy);
    append!(errors, [err])
    writedlm("errors.log", errors)

    print("Computed coefficients for N=$N \n")

end

plot(Φ, θs, rs, vals, ttl=L"u(x,y)") # plot
PyPlot.savefig("plane-wave-sol.png", dpi=500)
slice_plot(60, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("plane-wave-sol-slice.pdf")


###
# Convergence plot
###
errors = readdlm("errors.log")

ns = [sum(getMs(N)) for N in 20:10:150]
ps = 20:10:150
Plots.plot(ps, errors,
    # label=L"\mathrm{Zernike/Zernike \,\, annular \,\, (2 \,\,elements)}",
    linewidth=3,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    # xlabel=L"$\# \mathrm{Basis \; functions}$",
    xlabel=L"$N_p$",
    # ylim=[1e-15, 1e2],
    # xlim = [0, 3.6e4],
    legend=:none,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    color=3,
    gridlinewidth = 2
)
Plots.savefig("plane-wave-convergence.pdf")
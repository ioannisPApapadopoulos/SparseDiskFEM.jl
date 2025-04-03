using RadialPiecewisePolynomials
using PyPlot, Plots, DelimitedFiles
using SparseDiskFEM

"""
Section 6.5 "hp-refinement"

Domain is Ω = {0 ≤ r ≤ 1} and we are solving
    -Δ u(x,y) = 1/r
where the exact solution is u(x,y) = 1-√(x²+y²) = 1 - r.

In order to recover spectral convergence to the solution,
we generated a graded mesh towards the origin whilst simultaneously
considering higher order discretizations.
"""


ũ(r) = -r + 1
uₑ(r,θ) = ũ(r)
function uₑ_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    uₑ(r,θ)
end
f(r) = 1/r
# RHS in Cartesian coordinates
function rhs_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    f(r)
end

# hp-refinement
errors = []
for N in 4:45
    # Generate graded mesh
    points = [0.0;[2.0^(-n) for n in N:-0.5:0]]
    Nₕ = length(points)-1
    # Construct H¹ conforming disk FEM basis, truncation degree N
    Φ = ContinuousZernike(N, points);
    # Construct L² conforming disk FEM basis, truncation degree N
    Ψ = ZernikeBasis(N, points, 0, 0);
    # Analysis, compute coefficient vector for RHS
    fz = Ψ \ rhs_xy.(axes(Ψ,1));

    # Assemble the matrices
    A = stiffness_matrix(Φ); # <∇v, ∇u>, v, u ∈ Φ
    G = (Φ' * Ψ); # <v, u>, v ∈ Φ, u ∈ Ψ

    K = Matrix.(A); # <∇v, ∇u>
    Mf = G .* fz # <v, f>

    # Apply zero Dirichlet bcs
    zero_dirichlet_bcs!(Φ, [K...])
    zero_dirichlet_bcs!(Φ, Mf)

    # Solve over each Fourier mode separately
    u = K .\ Mf;
    (θs, rs, vals) = finite_plotvalues(Φ, u, N=100);
    vals_, err = inf_error(Φ, θs, rs, vals, uₑ_xy);
    push!(errors, err)
    writedlm("errors-hp-refinement.log", errors)

    print("Computed coefficients for N=$N, err=$err. \n")
end

# Plot solution
SparseDiskFEM.plot(Φ, θs, rs, vals, ttl=L"u(x,y)") # plot
PyPlot.savefig("hp-refinement-sol.png", dpi=500)
slice_plot(1, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("hp-refinement-sol-slice.pdf")


###
# Convergence plot
###
errors = readdlm("errors-hp-refinement.log")
Ns = 4:45
Plots.plot(Ns, errors,
    linewidth=3,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$N$",
    legend=:none,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    # xscale=:log10,
    yticks=[1e-14, 1e-12, 1e-10, 1e-8,1e-6,1e-4,1e-2,1e0],
    ylim=[1e-14,1e-1],
    color=1,
    gridlinewidth = 2
)
Plots.savefig("hp-refinement-convergence.pdf")
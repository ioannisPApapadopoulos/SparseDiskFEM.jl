using RadialPiecewisePolynomials
using PyPlot, Plots, DelimitedFiles
using SparseDiskFEM

"""
Section 6.5 "hp-refinement"

Domain is Ω = {0 ≤ r ≤ 1} and we are solving
    -Δ u(x,y) = r^{-3/2}
where the exact solution is u(x,y) = 4-4r^{1/2}.

In order to recover spectral convergence to the solution,
we generated a graded mesh towards the origin whilst simultaneously
considering higher order discretizations.
"""

ũ(r) = (4.0 - 4*sqrt(r))
uₑ(r,θ) = ũ(r)
function uₑ_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    uₑ(r,θ)
end
f(r) = 1/sqrt(r)^3
# RHS in Cartesian coordinates
function rhs_xy(xy)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    f(r)
end

g(r) = r*f(r)
using ClassicalOrthogonalPolynomials
import ClassicalOrthogonalPolynomials: plan_grid_transform
x,F = plan_grid_transform(legendre(0..1), 1000)
F*g.(x)

function hp_refinement_solve(points, N; Nc=100)
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
    (θs, rs, vals) = finite_plotvalues(Φ, u, N=Nc);
    vals_, err = inf_error(Φ, θs, rs, vals, uₑ_xy);
    return err, u, (θs, rs, vals, vals_)
end


## Convergence with graded mesh and p-refinement
errors_hp, dofs_hp = [], []
for N in 4:2:38
    # Generate graded mesh
    points = [0.0;[2.0^(-n) for n in (2*N):-1:0]]
    err, u, (θs, rs, vals, vals_) = hp_refinement_solve(points, N)
    push!(errors_hp, err)
    push!(dofs_hp, sum(length(u[1])))
    writedlm("errors-hp-refinement_hp.log", errors_hp)
    writedlm("errors-hp-refinement-dofs_hp.log", dofs_hp)
    print("Computed coefficients for N=$N, dofs=$(dofs_hp[end]), err=$err. \n")
end

## Convergence with graded mesh and fixed p=38
errors_h, dofs_h = [], []
for N in 4:2:38
    # Generate graded mesh
    points = [0.0;[2.0^(-n) for n in (2*N):-1:0]]
    err, u, (θs, rs, vals, vals_) = hp_refinement_solve(points, 38)
    push!(errors_h, err)
    push!(dofs_h, sum(length(u[1])))
    writedlm("errors-hp-refinement_h.log", errors_h)
    writedlm("errors-hp-refinement-dofs_h.log", dofs_h)
    print("Computed coefficients for N=$N, dofs=$(dofs_h[end]), err=$err. \n")
end

# Plot solution
N=38; points = [0.0;[2.0^(-n) for n in (2*N):-1:0]]
err, u, (θs, rs, vals) = hp_refinement_solve(points, N)
slice_plot(1, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("hp-refinement-sol-slice.pdf")
# SparseDiskFEM.plot(θs, rs, vals, ttl=L"u(x,y)") # plot
# PyPlot.savefig("hp-refinement-sol.png", dpi=500)

## Pure p-refinement
errors_p, dofs_p = [], []
points = [0.0;1.0]
for N in 4:100:1004
    # Generate graded mesh
    err, u, (θs, rs, vals, vals_) = hp_refinement_solve(points, N, Nc=N+100)
    push!(errors_p, err)
    push!(dofs_p, sum(length(u[1])))
    writedlm("errors-hp-refinement_p.log", errors_p)
    writedlm("errors-hp-refinement-dofs_p.log", dofs_p)
    print("Computed coefficients for N=$N, dofs=$(dofs_p[end]), err=$err. \n")
end


###
# Convergence plot
###
errors_hp, errors_h, errors_p = readdlm("errors-hp-refinement_hp.log"), 
    readdlm("errors-hp-refinement_h.log"),
    readdlm("errors-hp-refinement_p.log")
dofs_hp, dofs_h, dofs_p = readdlm("errors-hp-refinement-dofs_hp.log"),
    readdlm("errors-hp-refinement-dofs_h.log"),
    readdlm("errors-hp-refinement-dofs_p.log")

Plots.plot(dofs_hp, errors_hp,
    linewidth=3,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel="Degrees of freedom",
    legend=:topright,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    # xscale=:log10,
    yticks=[1e-10, 1e-8,1e-6,1e-4,1e-2,1e0],
    # ylim=[1e-15,1e-1],
    # color=1,
    gridlinewidth = 2,
    label="Graded mesh, p-refinement"
)
Plots.plot!(dofs_h, errors_h,
    linewidth=2,
    marker=:dtriangle,
    label="Graded mesh "* L"N_p=38")
Plots.plot!(dofs_p, errors_p,
    linewidth=2,
    marker=:square,
    label="Pure p-refinement")
Plots.savefig("hp-refinement-convergence.pdf")
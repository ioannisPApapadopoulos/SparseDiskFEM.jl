using RadialPiecewisePolynomials, LinearAlgebra
using PyPlot, Plots
using JLD
using SparseDiskFEM

"""
Section 6.2 "High frequency with a discontinuous Helmholtz coefficient"

Domain is Ω = {0 ≤ r ≤ 1} and we are solving
    (-Δ - λ(r))u(x,y) = f(x,y)
where the exact solution is not known. Thus we compare against reference solutions.

⚠ Please run high-frequency-SEM-soln.jl first to compute the spectral element
reference solution if "examples/data/high-frequency-SEM-ref-soln.jld" is not available.
"""

ρ = 0.5
# Negative Helmholtz parameter, in the indefinite Helmholtz regime
λ(r) = r ≤ ρ ? -80^2 : -90^2

# RHS
function f(xy)
    x, y = first(xy), last(xy)
    if x^2 + y^2 ≤ ρ^2
        return 2*sin(200*x)
    else
        return sin(100*y)
    end
end

# Construct endpoints for cells in the mesh
s = ρ^(-1/11)
points = [0; reverse([s^(-j) for j in 0:11])]
Nₕ = length(points)-1
N=200;
# Construct H¹ conforming disk FEM basis, truncation degreΔAe N=200
@time Φ = ContinuousZernike(N, points);
# Construct L² conforming disk FEM basis, truncation degree N=200
@time Ψ = ZernikeBasis(N, points, 0, 0);

x = axes(Ψ,1)
# Analysis, compute coefficient vector for RHS
fz = Ψ \ f.(x);
# Synthesis, evaluate discretized RHS and check the error
(θs, rs, vals) = finite_plotvalues(Ψ, fz, N=800);
vals_, err = inf_error(Ψ, θs, rs, vals, f); # Check inf-norm errors on the grid
err
# Plot the RHS
SparseDiskFEM.plot(Ψ, θs, rs, vals, ttl=L"f(x,y)")
PyPlot.savefig("high-frequency-rhs.png", dpi=500)
slice_plot(162, θs, rs, vals, points, ylabel=L"$f(x,y)$")
Plots.savefig("high-frequency-rhs-slice.pdf")

# Aseemble the matrices
@time Λ = piecewise_constant_assembly_matrix(Φ, λ); # <v, λ u>, v, u ∈ Φ
D = Derivative(axes(Φ,1))
@time A = ((D*Φ)' * (D*Φ));  # <∇v, ∇u>, v, u ∈ Φe
K = Matrix.(A .+ Λ);
@time G = (Φ' * Ψ);  # <v, u>, v ∈ Φ, u ∈ Ψ
Mf =  G .* fz; # <v, f>

# Apply zero Dirichlet bcs
zero_dirichlet_bcs!(Φ, K);
zero_dirichlet_bcs!(Φ, Mf);

# Reverse LU factors and spy plots
B = [Ks[end:-1:1, end:-1:1] for Ks in K];
LbUb = [lu(Bs, NoPivot()) for Bs in B];
Lb = [L.L for L in LbUb];
Ub = [U.U for U in LbUb];
L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
norm(K .- U.*L, Inf)

for (B, b) in zip((K, U, L), ("A", "U", "L"))
    Bₖ = B[350][:,:]
    Bₖ[abs.(Bₖ) .> 1e-15] .= 1
    Bₖ[abs.(Bₖ) .< 1e-15] .= 0
    Plots.spy(Bₖ, legend=:none, markersize=2)
    Plots.savefig("high-frequency-spy-m-175-$b.pdf")
end

# Solve over each Fourier mode seperately
u = K .\ Mf

# Plot the solution
(θs, rs, vals) = finite_plotvalues(Φ, u, N=800)
SparseDiskFEM.plot(Φ, θs, rs, vals, ttl=L"u(x,y)") # plot
PyPlot.savefig("high-frequency-sol.png", dpi=500)
slice_plot(162, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("high-frequency-sol-slice.pdf")


# Convergence against reference solutions. 
# vals_SEM_ref is the values computed via a spectral element method
# as implemented in high-frequency-SEM-soln.jl
(θs, rs, vals_fine) = finite_plotvalues(Φ, u, N=300)
vals_SEM_ref = JLD.load("examples/data/high-frequency-SEM-ref-soln.jld")["vals"]
# Error of O(1e-12) between FEM and SEM reference solutions
maximum(norm.(vals_fine .- vals_SEM_ref, Inf))
    

errors_ref = []
errors_fine = []
for N in 20:10:200
    # Ms is the number of basis functions in each Fourier mode
    # up to degree n.
    Ms = getNs(N)
    
    # Truncate the matrices at degree N
    Kn = [K[j][1:(n-1)*Nₕ, 1:(n-1)*Nₕ] for (n,j) in zip(Ms,1:lastindex(K))];
    Mfn = [Mf[j][1:(n-1)*Nₕ] for (n,j) in zip(Ms,1:lastindex(Mf))];# right-hand side
    
    # Solve over each Fourier mode seperately
    un = Kn .\ Mfn;
    (θs, rs, vals) = finite_plotvalues(Φ, un, N=300);

    append!(errors_ref, [maximum(norm.(vals .- vals_SEM_ref, Inf))])
    append!(errors_fine, [maximum(norm.(vals .- vals_fine, Inf))])
    writedlm("errors_ref-high-frequency.log", errors_ref)
    writedlm("errors_fine-high-frequency.log", errors_fine)
    
    print("Computed coefficients for N=$N \n")

end

###
# Convergence plot
###
errors_ref = readdlm("errors_ref-high-frequency.log")
errors_fine = readdlm("errors_fine-high-frequency.log")
errors_fine[end] = NaN
# ns = [sum(getMs(N)) for N in 20:10:150]
ps = 20:10:200
Plots.plot(ps, [errors_ref errors_fine],
    label=["SEM reference" "hp-FEM reference"],
    linewidth=3,
    markershape=[:circle :dtriangle],
    linestyle=[:solid :dash],
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    # xlabel=L"$\# \mathrm{Basis \; functions}$",
    xlabel=L"$N_p$",
    # ylim=[1e-15, 1e2],
    # xlim = [0, 3.6e4],
    legend=:topright,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize = 10,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    # color=3,
    gridlinewidth = 2
)
Plots.savefig("high-frequency-convergence.pdf")
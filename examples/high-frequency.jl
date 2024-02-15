using RadialPiecewisePolynomials, LinearAlgebra
using PyPlot, Plots
using JLD
using SparseDiskFEM

ρ = 0.5
λ(r) = r ≤ ρ ? -80^2 : -90^2

function f(xy)
    x, y = first(xy), last(xy)
    if x^2 + y^2 ≤ ρ^2
        return 2*sin(200*x)
    else
        return sin(100*y)
    end
end

s = ρ^(-1/11)
points = [0; reverse([s^(-j) for j in 0:11])]
K = length(points)-1
N=200;
@time Φ = ContinuousZernike(N, points);
@time Ψ = ZernikeBasis(N, points, 0, 0);

x = axes(Ψ,1)

fz = Ψ \ f.(x);
(θs, rs, vals) = finite_plotvalues(Ψ, fz, N=800);
vals_, err = inf_error(Ψ, θs, rs, vals, f); # Check inf-norm errors on the grid
err
SparseDiskFEM.plot(Ψ, θs, rs, vals, ttl=L"f(x,y)")
PyPlot.savefig("high-frequency-rhs.png", dpi=500)
slice_plot(162, θs, rs, vals, points, ylabel=L"$f(x,y)$")
Plots.savefig("high-frequency-rhs-slice.pdf")

# Solve Helmholtz equation in weak form
# @time M = Φ' * (V.(axes(Φ,1)) .* Φ); # list of assembly matrices for each Fourier mode
@time Λ = piecewise_constant_assembly_matrix(Φ, λ);
D = Derivative(axes(Φ,1))
@time Δ = ((D*Φ)' * (D*Φ)); # list of stiffness matrices for each Fourier mode
A = Matrix.(Δ .+ Λ);
@time G = (Φ' * Ψ);
Mf =  G .* fz; # right-hand side
zero_dirichlet_bcs!(Φ, A); # bcs
zero_dirichlet_bcs!(Φ, Mf); # bcs


B = [As[end:-1:1, end:-1:1] for As in A];
LbUb = [lu(Bs, NoPivot()) for Bs in B];
Lb = [L.L for L in LbUb];
Ub = [U.U for U in LbUb];
L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
norm(A .- U.*L, Inf)

for (B, b) in zip((A, U, L), ("A", "U", "L"))
    Bₖ = B[350][:,:]
    Bₖ[abs.(Bₖ) .> 1e-15] .= 1
    Bₖ[abs.(Bₖ) .< 1e-15] .= 0
    Plots.spy(Bₖ, legend=:none, markersize=2)
    Plots.savefig("high-frequency-spy-m-175-$b.pdf")
end

# Solve over each Fourier mode seperately
u = A .\ Mf
# u = L .\ (U .\ Mf)

(θs, rs, vals) = finite_plotvalues(Φ, u, N=800)
SparseDiskFEM.plot(Φ, θs, rs, vals, ttl=L"u(x,y)") # plot
PyPlot.savefig("high-frequency-sol.png", dpi=500)
slice_plot(162, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("high-frequency-sol-slice.pdf")

(θs, rs, vals_fine) = finite_plotvalues(Φ, u, N=300)
vals_ref = JLD.load("high-frequency-soln.jld")["vals"]

maximum(norm.(vals_fine .- vals_ref, Inf))
    

errors_ref = []
errors_fine = []
for N in 20:10:200
    Ms = getNs(N)
    
    An = [A[j][1:(n-1)*K, 1:(n-1)*K] for (n,j) in zip(Ms,1:lastindex(A))];
    Mfn = [Mf[j][1:(n-1)*K] for (n,j) in zip(Ms,1:lastindex(Mf))];# right-hand side
    
    un = An .\ Mfn;
    (θs, rs, vals) = finite_plotvalues(Φ, un, N=300);

    append!(errors_ref, [maximum(norm.(vals .- vals_ref, Inf))])
    append!(errors_fine, [maximum(norm.(vals .- vals_fine, Inf))])
    writedlm("errors_ref-high-frequency.log", errors_ref)
    writedlm("errors_fine-high-frequency.log", errors_fine)
    
    print("Computed coefficients for N=$N \n")

end

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
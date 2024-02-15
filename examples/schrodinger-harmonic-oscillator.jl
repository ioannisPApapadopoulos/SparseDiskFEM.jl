using RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials, LinearAlgebra
using PyPlot, Plots, LaTeXStrings, JLD
using SparseDiskFEM

"""
Section 6.3 "Time-dependent Schrödinger equation"

Domain is Ω = {0 ≤ r ≤ 50} and we are solving
    i ∂ₜ u(x,y,t) = (-Δ + r²) u(x,y,t), u⁽⁰⁾(x, y) = ψ_{20,21}(x, y)
on the disk with zero Dirichlet bcs at r=50, 

where ψ_{20,21} is an eigenfunction of (-Δ + r²).

The exact solution is u(x,y,t) =  exp(−i E_{20,21} t) ψ_{20,21}(x, y) where
E_{20,21} is the eigenvalue corresponding to ψ_{20,21}(x, y).

"""

# Eigenvalue parameters
nx, ny = 20, 21
E = 2*(nx+ny+1)
# Eigenfunctions of (-Δ + r²)
function ψa(xy, t)
    x, y = first(xy), last(xy)
    H = Normalized(Hermite())
    exp(-E*im*t) * H[x,nx+1] * H[y,ny+1] * exp(-(x^2+y^2)/2)
end

# Initial state at t=0
function u0(xy)
    ψa(xy, 0)
end

# Endpoints of cells in the mesh
points = [0; [50*1.2^(-n) for n in 15:-1:0]]; K = length(points)-1;
# Construct H¹ conforming disk FEM basis, truncation degree N=100
N=100; Φ = ContinuousZernike(N, points);

V(r²) = r² # quadratic well

# Analysis, compute coefficient vector of initial state
u0c_F = Φ \ u0.(axes(Φ,1))
# Synthesis, evaluate discretized initial state and check the error
(θs, rs, vals) = finite_plotvalues(Φ, u0c_F, N=300)
vals_, err = inf_error(Φ, θs, rs, vals, u0) # Check inf-norm errors on the grid
err
# Plot initial state over all the cells.
SparseDiskFEM.plot(Φ, θs, rs, vals, ttl=L"u^{(0)}(x,y)")
PyPlot.savefig("schrodinger-u0-full.png", dpi=500)
# Plot initial state over the first 8 cells starting at r=0.
SparseDiskFEM.plot(Φ, θs, rs, vals, K=8, ttl=L"u^{(0)}(x,y)")
PyPlot.savefig("schrodinger-u0.png", dpi=500)
# 1D slice plot over the first 8 cells starting at r=0.
slice_plot(50, θs[1:8], rs[1:8], vals[1:8], points[1:9], ylabel=L"$u^{(0)}(x,y)$")
Plots.savefig("schrodinger-u0-slice.pdf")

# Solve Helmholtz equation in weak form
@time M = Φ' * Φ; # <v, u>, v, u ∈ Φ
@time wM = Φ' * (V.(axes(Φ,1)) .* Φ); # <v, V(r²) u>, v, u ∈ Φ
D = Derivative(axes(Φ,1));
@time nΔ = (D*Φ)' * (D*Φ); # <∇v, ∇u>, v, u ∈ Φ

# Setup of time loop via a Crank-Nicolson time discretization
us = repeat([[]], 9)
# Number of timesteps from t=0 to t=2π/E
kTs = [1, 5, 10, 60, 100, 400, 700, 1000, 1300]
for (kT, i) in zip(kTs, 1:lastindex(kTs))
    δt = 2π / E / kT # Step size
    # Crank-Nicolson matrices
    Ap = Matrix.(2 .*M .+ (im * δt) .* (nΔ .+ wM));
    An = Matrix.(2 .*M .- (im * δt) .* (nΔ .+ wM));

    zero_dirichlet_bcs!(Φ, Ap); # bcs
    us[i] = Any[u0c_F];
    u = u0c_F;

    # reverse complex-valued LU factorization
    B = [A[end:-1:1, end:-1:1] for A in Ap];
    LbUb = [lu(Bs, NoPivot()) for Bs in B];
    Lb = [L.L for L in LbUb];
    Ub = [U.U for U in LbUb];
    # norm(B .- Lb.*Ub, Inf)
    L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
    U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
    # norm(Ap .- U.*L, Inf)

    # for loop for time-stepping
    @time for its in 1:kT
        # explicit time half-step
        fu = (An .* u)
        zero_dirichlet_bcs(Φ, fu);
        # implicit time half-step
        u = L .\ (U .\ fu)
        append!(us[i], [u])
        print("Time step: $its \n")
    end
end
# save solution coefficients
JLD.save("schrodinger-us.jld", "us", us)


# Compute the error at the final time step, t=2π/E.
# Note the error is always largest at the final time step
errs = []
for i = 1:9
    its = lastindex(us[i])
    # Sythesis (expansion) for real and complex-valued coefficients
    (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[i][its]), N=150);
    (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[i][its]), N=150);
    t = 2π/E

    # tdisplay = round(t, digits=4)
    # SparseDiskFEM.plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
    # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

    # Compute the ℓ^∞ error
    ua(xy) = ψa(xy,t)
    vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua)
    append!(errs, err)
    print("error: $(errs[end]) \n")
    writedlm("errors-schrodinger-harmonic-oscillator.log", errs)
end

###
# Convergence plots
###
δts = 2π / E ./ kTs
# Expected second-order convergence
second_order_rate = δts.^2 ./ δts[2]^2 / 3 * errs[1]
Plots.plot(δts, Float64.(hcat(errs, second_order_rate)),
    label=["" L"O(\delta t^2)"],
    linewidth=[3 2],
    markershape=[:circle :none],
    linestyle=[:solid :dash],
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\delta t$",
    legend=:bottomright,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize = 13,
    yscale=:log10,
    xscale=:log10,
    yticks=[1e-6, 1e-4, 1e-2, 1e0],
    xticks=[1e-4,1e-3,1e-2,1e-1],
    xlim=[5e-5,1e-1],
    ylim=[1e-6,1e0],
    gridlinewidth = 2,
    margin=4Plots.mm,
)
Plots.savefig("schrodinger-convergence.pdf")

###
# Spy plots of reverse complex-valued LU factors
###
δt = 2π / E / kTs[1]
Ap = Matrix.(2 .*M .+ (im * δt) .* (nΔ .+ wM));
zero_dirichlet_bcs!(Φ, Ap); # bcs
B = [A[end:-1:1, end:-1:1] for A in Ap];
LbUb = [lu(Bs, NoPivot()) for Bs in B];
Lb = [L.L for L in LbUb];
Ub = [U.U for U in LbUb];
L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
norm(Ap .- U.*L, Inf)
for (B, b) in zip((Ap, U, L), ("A", "U", "L"))
    Bₖ = B[140][:,:]
    Bₖ[abs.(Bₖ) .> 1e-15] .= 1
    Bₖ[abs.(Bₖ) .< 1e-15] .= 0
    Plots.spy(Bₖ, legend=:none, markersize=1.5)
    Plots.savefig("schrodinger-spy-m-70-$b.pdf")
end

###
# Measure conservation of energy (L^2-norm of iterates should stay roughly constant)
###
l2_norms = []
for its in 1:1301
    append!(l2_norms, [sqrt(abs(sum(adjoint.(us[9][its]) .* (M .* us[9][its]))))])
    print("Time step: $its \n")
end

# Difference in L^2-norm with initial state
l2_diff = abs.(l2_norms .- l2_norms[1])
Plots.plot((2:1301), l2_diff[2:1301],
    linewidth=2,
    ylabel=L"$|\Vert u^{(k)} \Vert_{L^2(\Omega)} - \Vert u^{(0)}  \Vert_{L^2(\Omega)}| $",
    xlabel=L"$k$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    markersize = 2,
    marker=:circle,
    legend=:none,
    margin=4Plots.mm
)
Plots.savefig("schrodinger-l2norm.pdf")

###
# A one-cell discretization would require Nₚ=700 to capture the
# the right-hand side to the same error
###
Ψ = ZernikeBasis(700, [0.0;50.0], 0, 0);
u0z = Ψ \ u0.(axes(Ψ,1));
(θs, rs, vals) = finite_plotvalues(Ψ, u0z, N=1000);
vals_, err = inf_error(Ψ, θs, rs, vals, u0)
err
SparseDiskFEM.plot(Ψ, θs, rs, vals,  ttl=L"u^{(0)}(x,y)")

# Φ = ContinuousZernike(700, [0.0;50.0]);
# u0ϕ = Φ \ u0.(axes(Ψ,1));
# (θs, rs, vals) = finite_plotvalues(Φ, u0ϕ);
# vals_, err = inf_error(Φ, θs, rs, vals, u0)
# err

# # For making a movie of the time evolution and measuring the error over all the
# # choices of the time step
# errs = []
# for kT in kTs
#     i = 1;
#     for its in 1:10:lastindex(us)
#         i += 1
#         (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[its]), N=100, K=Kp);
#         (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[its]), N=100, K=Kp);
#         t = ((its-1)*δt)
#         tdisplay = round(t, digits=4)

#         # SparseDiskFEM.plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
#         # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

#         ua(xy) = ψa(xy,t)
#         vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua, K=Kp)
#         append!(errs, err)
#         print("error: $(errs[end]) \n")
#         # writedlm("examples/plots-harmonic-oscillator/errors.log", errs)
#     end
# end
# ffmpeg -framerate 8 -i %d.png -pix_fmt yuv420p out.mp4
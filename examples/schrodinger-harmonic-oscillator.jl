using RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials, LinearAlgebra
using PyPlot, Plots, LaTeXStrings
using SparseDiskFEM # plotting routines

"""
Solving Schroedinger's equation:
    i ∂ₜ u(x,y,t) = (-Δ + r²) u(x,y,t)

on the disk with zero Dirichlet bcs at r=40.

Here we pick:
    u₀(x,y) = eigenfunction. 
"""

function zero_dirichlet_bcs(Φ::ContinuousZernike{T}, Mf::AbstractVector{<:AbstractVector}) where T
    @assert length(Mf) == 2*Φ.N-1
    Fs = Φ.Fs #_getFs(Φ.N, Φ.points)
    zero_dirichlet_bcs.(Fs, Mf)
end

function zero_dirichlet_bcs(Φ::ContinuousZernikeMode{T}, Mf::AbstractVector) where T
    points = Φ.points
    K = length(points)-1
    if !(first(points) ≈  0)
        Mf[1] = zero(T)
        Mf[K+1] = zero(T)
    else
        Mf[K] = zero(T)
    end
end

nx, ny = 20, 21
E = 2*(nx+ny+1)
function ψa(xy, t)
    x, y = first(xy), last(xy)
    H = Normalized(Hermite())
    exp(-E*im*t) * H[x,nx+1] * H[y,ny+1] * exp(-(x^2+y^2)/2)
end

function u0(xy)
    ψa(xy, 0)
end

points = [0; [50*1.2^(-n) for n in 15:-1:0]]; K = length(points)-1;
Kp = 8
# points = [0.0; 40.0]; K = length(points)-1
N=100; Φ = ContinuousZernike(N, points);


V(r²) = r² # quadratic well


u0c_F = Φ \ u0.(axes(Φ,1))
(θs, rs, vals) = finite_plotvalues(Φ, u0c_F, N=300)
vals_, err = inf_error(Φ, θs, rs, vals, u0) # Check inf-norm errors on the grid
err
plot(Φ, θs, rs, vals, K=Kp, ttl=L"u^{(0)}(x,y)")
PyPlot.savefig("schrodinger-u0.png", dpi=500)
plot(Φ, θs, rs, vals, ttl=L"u^{(0)}(x,y)")
PyPlot.savefig("schrodinger-u0-full.png", dpi=500)
slice_plot(50, θs, rs, vals, points[1:9], ylabel=L"$u^{(0)}(x,y)$")
Plots.savefig("schrodinger-u0-slice.pdf")

# Solve Helmholtz equation in weak form
@time M = Φ' * Φ;
@time wM = Φ' * (V.(axes(Φ,1)) .* Φ); # list of weighted mass matrices for each Fourier mode
D = Derivative(axes(Φ,1));
@time nΔ = (D*Φ)' * (D*Φ); # list of stiffness matrices for each Fourier mode
# R = Φ'*Z;

us = [[],[],[],[],[],[],[],[],[]]
kTs = [1, 5, 10, 60, 100, 400, 700, 1000, 1300]
for (kT, i) in zip(kTs, 1:lastindex(kTs))
# kT = 1
# i = 1
    δt = 2π / E / kT
    Ap = Matrix.(2 .*M .+ (im * δt) .* (nΔ .+ wM));
    An = Matrix.(2 .*M .- (im * δt) .* (nΔ .+ wM));

    zero_dirichlet_bcs!(Φ, Ap); # bcs
    us[i] = Any[u0c_F];
    u = u0c_F;


    B = [A[end:-1:1, end:-1:1] for A in Ap];
    LbUb = [lu(Bs, NoPivot()) for Bs in B];
    Lb = [L.L for L in LbUb];
    Ub = [U.U for U in LbUb];
    # norm(B .- Lb.*Ub, Inf)
    L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
    U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
    # norm(Ap .- U.*L, Inf)


    @time for its in 1:kT
        fu = (An .* u)
        zero_dirichlet_bcs(Φ, fu);
        # u = Ap .\ fu
        u = L .\ (U .\ fu)
        append!(us[i], [u])
        print("Time step: $its \n")
    end
end
# using JLD
# JLD.save("schrodinger-us.jld", "us", us)


errs = []
for i = 1:9
    its = lastindex(us[i])
    (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[i][its]), N=150);
    (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[i][its]), N=150);
    t = 2π/E
    # tdisplay = round(t, digits=4)

    # plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
    # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

    ua(xy) = ψa(xy,t)
    vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua)
    append!(errs, err)
    print("error: $(errs[end]) \n")
    # writedlm("examples/plots-harmonic-oscillator/errors.log", errs)
end

δts = 2π / E ./ kTs
rate = δts.^2 ./ δts[2]^2 / 3 * errs[1]
Plots.plot(δts, Float64.(hcat(errs, rate)),
    label=["" L"O(\delta t^2)"],
    linewidth=[3 2],
    markershape=[:circle :none],
    linestyle=[:solid :dash],
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\delta t$",
    # ylim=[1e-15, 1e2],
    # xlim = [0, 3.6e4],
    legend=:bottomright,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize = 13,
    yscale=:log10,
    xscale=:log10,
    yticks=[1e-6, 1e-4, 1e-2, 1e0],
    xticks=[1e-4,1e-3,1e-2,1e-1],
    xlim=[5e-5,1e-1],
    ylim=[1e-6,1e0],
    # color=3,
    gridlinewidth = 2,
    margin=4Plots.mm,
)
Plots.savefig("schrodinger-convergence.pdf")

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

# errs = []
# for kT in kTs
#     i = 1;
#     for its in 1:10:lastindex(us)
#         i += 1
#         (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[its]), N=100, K=Kp);
#         (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[its]), N=100, K=Kp);
#         t = ((its-1)*δt)
#         tdisplay = round(t, digits=4)

#         # plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
#         # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

#         ua(xy) = ψa(xy,t)
#         vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua, K=Kp)
#         append!(errs, err)
#         print("error: $(errs[end]) \n")
#         # writedlm("examples/plots-harmonic-oscillator/errors.log", errs)
#     end
# end

l2_norms = []
# h1_norms = []
# H = nΔ .+ M;
for its in 1:1301
    append!(l2_norms, [sqrt(abs(sum(adjoint.(us[9][its]) .* (M .* us[9][its]))))])
    # append!(h1_norms, [sqrt(abs(sum(adjoint.(us[its]) .* (H .* us[its]))))])
    print("Time step: $its \n")
end


l2_diff = abs.(l2_norms .- l2_norms[1])
Plots.plot((2:1301), l2_diff[2:1301],
    linewidth=2,
    ylabel=L"$|\Vert u^{(k)} \Vert_{L^2(\Omega)} - \Vert u^{(0)}  \Vert_{L^2(\Omega)}| $",
    xlabel=L"$k$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    # linestyle = :dot,
    markersize = 2,
    marker=:circle,
    legend=:none,
    # size=(850,400),
    margin=4Plots.mm
)
Plots.savefig("schrodinger-l2norm.pdf")
# ffmpeg -framerate 8 -i %d.png -pix_fmt yuv420p out.mp4


Ψ = ZernikeBasis(700, [0.0;50.0], 0, 0);
u0z = Ψ \ u0.(axes(Ψ,1));
(θs, rs, vals) = finite_plotvalues(Ψ, u0z, N=1000);
vals_, err = inf_error(Ψ, θs, rs, vals, u0)
err
plot(Ψ, θs, rs, vals,  ttl=L"u^{(0)}(x,y)")

Φ = ContinuousZernike(700, [0.0;50.0]);
u0ϕ = Φ \ u0.(axes(Ψ,1));
(θs, rs, vals) = finite_plotvalues(Φ, u0ϕ);
vals_, err = inf_error(Φ, θs, rs, vals, u0)
err
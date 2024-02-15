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

function ψ(xy)
    x, y = first(xy), last(xy)
    H = Normalized(Hermite())
    H[x,nx+1] * H[y,ny+1] * exp(-(x^2+y^2)/2)
end

points = [0; [50*1.2^(-n) for n in 15:-1:0]]; K = length(points)-1;
Kp = 8
# points = [0.0; 40.0]; K = length(points)-1
N=100; Φ = ContinuousZernike(N, points);


V(r²) = r² # quadratic well

# Solve Helmholtz equation in weak form
@time M = Φ' * Φ;
@time wM = Φ' * (V.(axes(Φ,1)) .* Φ); # list of weighted mass matrices for each Fourier mode
D = Derivative(axes(Φ,1));
@time nΔ = (D*Φ)' * (D*Φ); # list of stiffness matrices for each Fourier mode

A = Matrix.(nΔ .+ wM);
zero_dirichlet_bcs!(Φ, A)
M = Matrix.(M)
zero_dirichlet_bcs!(Φ, [M...])

# nx, ny = 4, 0
# E = 2*(nx+ny+1)

Es = eigen.(A, M);
evs = [E.values[2:end] for E in Es];


# i=2; ev, ef = eigen(A[i], M[i])
# c2 = ef[:,3];
# i=6; ev, ef = eigen(A[i], M[i])
# c3 = ef[:,2];

# u = [zeros.(axes.(M,1))...]
# u[2]=c2/2; #u[3]=c3;
# u[6]=c3 * sqrt(3)/2
# (θs, rs, vals) = finite_plotvalues(Φ, u, N=150, K=3);
# _, err = inf_error(Φ, θs, rs, vals, ψ, K=3);
# err
# SparseDiskFEM.plot(Φ, θs, rs, vals, K=3)

# Ψ = ZernikeBasis(N, points, 0, 0)
# v = Ψ \ ψ.(axes(Ψ,1))
# s = transpose.(v) .* (M .* v)



import RadialPiecewisePolynomials: findblockindex, blockedrange, Fill, oneto
tEs = 2*Vector(Vcat((Fill.(1:100, 1:100))...))
bl = [findblockindex(blockedrange(oneto(∞)), j) for j in 1:sum(1:N)]
ℓ = [bl[j].I[1]-1 for j in 1:sum(1:N)] # degree
k = [bl[j].α[1] for j in 1:sum(1:N)]   # index of degree
ms = [iseven(ℓ[j]) ? k[j]-isodd(k[j]) : k[j]-iseven(k[j]) for j in 1:sum(1:N)] # m-mode
js = isodd.(k .+ ℓ)
deg = (ℓ .- ms) .÷ 2 .+ 1

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
    # linewidth=0,
    linetype=:scatter,
    ylabel=L"$|E_h - E|/E$",
    xlabel=L"$E$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    # linestyle = :dot,
    markersize = 2,
    marker=:circle,
    legend=:none,
    ylim=[1e-17,1e0],
    yticks=[1e-15,1e-10,1e-5,1e0],
    # linestyle=:none,
    # size=(850,400),
    # margin=4Plots.mm
    yscale=:log10,
)
Plots.savefig("schrodinger-eigs.pdf")

# (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[i][its]), N=150);
# (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[i][its]), N=150);
# t = 2π/E

# plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
# PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")



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
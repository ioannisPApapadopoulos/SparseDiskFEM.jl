using RadialPiecewisePolynomials, PiecewiseOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using SparseDiskFEM, Plots, PyPlot # plotting routines
import ForwardDiff: derivative
using DelimitedFiles

ρ, N, Nz = 0.5, 60, 60
points = [0;ρ;1]
Nₕ = length(points) - 1

# Disk FEM basis 
Φ = FiniteContinuousZernike(N, points);
Ψ = FiniteZernikeBasis(N, points, 0, 0);

r = range(-1, 1; length=3)
Nzₕ = length(r)-1

# Interval FEM basis
P = ContinuousPolynomial{0}(r)
Q = ContinuousPolynomial{1}(r)

λ₀, λ₁ = 1e-2, 5e1
λ(r) = r ≤ ρ ? λ₀ : λ₁
ũ(r) = r ≤ ρ ? (λ₀*r^2/4 + (λ₁ - λ₀)*ρ^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(ρ)) : (λ₁*r^2/4 - λ₁/4 + (λ₀ - λ₁)/2*ρ^2*log(r))

k = 5
uₑ(r,θ,z) = cos(k*(r*cos(θ))) * ũ(r) * cos(k*z) * (1-z^6)
function uₑ_xyz(xy,z)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    uₑ(r,θ,z)
end

# Use ForwardDiff to compute the RHS
Δuₑ(r,θ,z) =
    (derivative(r->derivative(r->uₑ(r,θ,z), r),r) 
    + derivative(r->uₑ(r,θ,z), r)/r 
    + derivative(θ->derivative(θ->uₑ(r,θ,z), θ),θ)/r^2
    + derivative(z->derivative(z->uₑ(r,θ,z), z),z) 
    )

rhs(r,θ,z) =  -Δuₑ(r,θ,z) + λ(r)*uₑ(r,θ,z)

# RHS in Cartesian coordinates
function rhs_xyz(xy,z)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs(r,θ,z)
end

# Expand in disk basis for each point on the interval grid
𝐳 = ClassicalOrthogonalPolynomials.grid(P, Nzₕ*Nz)
v𝐳 = vec(𝐳)
X = disk_tensor_transform(Ψ, v𝐳, rhs_xyz, N);

# Expand in interval basis
PP = plan_transform(P, (Block(Nz), sum(1:N)), 1)
# Fs is the list of matrices of expansion coefficients for the RHS.
Fs = [(PP * reshape(X[k]', lastindex(𝐳)÷Nzₕ, Nzₕ, sum(1:N)))' for k in 1:Nₕ];


# Evaluation points
# 𝐳p = [-1.0;reverse(vec(𝐳[:,end:-1:1]));1.0]
𝐳p = range(-1,1,100);

# Evaluate the expansion and check the error
vals_f, rs, θs, vals_, errs = synthesis_error_transform(Ψ, P, Fs, 𝐳p, rhs_xyz, N, Nz);
maximum(errs)

# Assemble the matrices
D = Derivative(axes(Q,1))
pA = ((D*Q)' * (D*Q))
pM = Q' * Q
pG = (Q'*P)[Block.(1:Nz), Block.(1:Nz)];

D = Derivative(axes(Φ,1));
A = (D*Φ)' * (D*Φ);
Λ = piecewise_constant_assembly_matrix(Φ, λ);
M = Φ' * Φ;
G = Φ' * Ψ;

Λ = Matrix.(Λ);
M = Matrix.([M...]);
K = Matrix.(A .+  Λ);
zero_dirichlet_bcs!(Φ, M);
zero_dirichlet_bcs!(Φ, Λ);
zero_dirichlet_bcs!(Φ, K);

Us, avgJs = [], []

zFs = []
for i in 1:lastindex(𝐳)
    append!(zFs, [modaltrav_2_list(Ψ, [Fs[k][:,i] for k in 1:Nₕ])])
end

vals_u, θs, rs, errors = [], [], [], []

for n in 5 : 5 : N
    ns = getNs(n)

    tpA = Matrix(pA[Block.(1:n), Block.(1:n)]);
    # Reverse Cholesky
    tpA[:,1] .= 0; tpA[1,:] .= 0; tpA[1,1] = 1.0;
    tpA[:,Nzₕ+1] .= 0; tpA[Nzₕ+1,:] .= 0; tpA[Nzₕ+1,Nzₕ+1] = 1.0;

    rtpA = tpA[end:-1:1, end:-1:1]
    pL = cholesky(Symmetric(rtpA)).L
    pL = pL[end:-1:1, end:-1:1]
    @assert pL * pL' ≈ tpA

    pGn = Matrix(pG[Block.(1:n), Block.(1:n)])

    # Compute spectrum
    tpM = Matrix(pM[Block.(1:n), Block.(1:n)])
    tpM[:,1] .= 0; tpM[1,:] .= 0; tpM[1,1] = 1.0;
    tpM[:,Nzₕ+1] .= 0; tpM[Nzₕ+1,:] .= 0; tpM[Nzₕ+1,Nzₕ+1] = 1.0;

    B = -(pL \ (pL \ tpM)') # = L⁻¹ pM L⁻ᵀ
    c, d = eigmin(B), eigmax(B)

    Kn = [K[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(K))];
    Mn = [M[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(M))];
    Λn = [Λ[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(Λ))];
    Gn = [G[j][1:(N-1)*Nₕ, 1:N*Nₕ] for (N,j) in zip(ns,1:lastindex(G))];
    
    
    # Reverse Cholesky
    rK = [Ks[end:-1:1, end:-1:1] for Ks in Kn];
    Lb = [cholesky(Symmetric(rKs)).L for rKs in rK];
    L = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
    @assert norm(Kn .- L.*transpose.(L), Inf) < 1e-10

    
    zFsn = [[zFs[iz][j][1:n*Nₕ] for (n,j) in zip(ns,1:lastindex(zFs[iz]))] for iz in 1:lastindex(zFs)]
    nUs, Js = AbstractMatrix{Float64}[], []
    for i in 1:2n-1
        # Compute spectrum
        B = (L[i] \ (L[i] \ Mn[i])') # = L⁻¹ pM L⁻ᵀ
        a, b = eigmin(B), eigmax(B)

        γ = (c-a)*(d-b)/((c-b)*(d-a))
        append!(Js, [Int(ceil(log(16γ)*log(4/1e-15)/π^2))])

        # weak form for RHS
        fp = zeros(size(Gn[i],2), size(pGn,2))
        for j in 1:n fp[:,j] = zFsn[j][i] end
        F_rhs = Matrix(Gn[i])*fp*pGn'  # RHS <f,v>
        F_rhs[Nₕ, :] .= 0; # disk bcs
        F_rhs[:, 1] .=0; F_rhs[:, Nzₕ+1] .= 0; # interval bcs

        X = adi(Mn[i], -tpM, Kn[i], tpA, F_rhs, a, b, c, d, tolerance=1e-15)

        U = (pL' \ (pL \ X'))'
        append!(nUs, [U])
        
    end
    print("n = $n, completed ADI loops.")
    append!(Us, [nUs])
    append!(avgJs, [sum(Js)/length(Js)])
    vals_u, rs, θs, vals_u_, errs_u = synthesis_error_transform(Φ, Q, nUs, 𝐳p, uₑ_xyz, n, n)
    append!(errors, [maximum(errs_u)])
    print("   Computed ℓ-∞ error.\n")
end

#### Plotting

### RHS
# Plotting routines
iz= 30
zval = round(𝐳p[iz], digits=2)

# Disk slice
SparseDiskFEM.plot(Ψ, θs, rs, vals_f[iz], ttl=L"f(x,y,%$zval)")
PyPlot.savefig("adi-rhs-2dslice.png", dpi=500)
# z slice
θval = round(θs[1][iz], digits=4)
SparseDiskFEM.zplot(iz, rs, 𝐳p, vals_f, ttl=L"f(x,y,z)", xlabel=L"r", ylabel=L"z", title=L"\theta = %$(θval)")
PyPlot.savefig("adi-rhs-zslice.png", dpi=500)
# save for MATLAB 3D Plot.
write_adi_vals(𝐳p, rs, θs, vals_f)
# 1D slice
# SparseDiskFEM.slice_plot(iz, θs, rs, vals_f[iz], points, ylabel=L"$f(x,y,%$zval)$")
# Plots.savefig("adi-rhs-1dslice2.pdf")

## Solution
SparseDiskFEM.plot(Φ, θs, rs, vals_u[iz], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("adi-sol-2dslice.png", dpi=500)
θval = round(θs[1][iz], digits=4)
SparseDiskFEM.zplot(iz, rs, 𝐳p, vals_u, ttl=L"u(x,y,z)", xlabel=L"r", ylabel=L"z", title=L"\theta = %$(θval)")
PyPlot.savefig("adi-sol-zslice.png", dpi=500)
write_adi_vals(𝐳p, rs, θs, vals_u)
# SparseDiskFEM.slice_plot(iz, θs, rs, vals_u[iz], points, ylabel=L"$u(x,y,%$zval)$")
# Plots.savefig("adi-sol-1dslice.pdf")


errors = readdlm("errors-adi.log")
avgJs = readdlm("Js-adi.log")

ps = 5:5:N
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
    ylim=[1e-15,8],
    # color=3,
    gridlinewidth = 2
)
Plots.savefig("adi-convergence.pdf")

m = (avgJs[end]-avgJs[3])/(log(ps[end])-log(ps[3]))
fit = m*log.(ps) .- m*log(ps[end]) .+ avgJs[end]
Plots.plot(ps, [Float64.(avgJs) fit],
    linewidth=[3 2],
    markershape=[:circle :none],
    linestyle=[:solid :dash],
    markersize=5,
    label =["" L"O(\log \, N_p)"],
    ylabel=L"$\mathrm{avg.} l_{\mathrm{max}}$",
    xlabel=L"$N_p$",
    # legend=:none,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=12,
    # yscale=:log10,
    # xscale=:log10,
    # yticks=[1e-15, 1e-10, 1e-5, 1e0],
    # ylim=[1e-15,8],
    # color=3,
    gridlinewidth = 2
)
Plots.savefig("adi-Js.pdf")

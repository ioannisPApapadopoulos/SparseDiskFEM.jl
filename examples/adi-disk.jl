using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using SparseDiskFEM, Plots, PyPlot # plotting routines
import ForwardDiff: derivative
using DelimitedFiles

ρ = 0.5
N = 60
# s = ρ^(-1/5)
# points = [0; reverse([s^(-j) for j in 0:5])]
points = [0;ρ;1]
Nₕ = length(points) - 1
@time Φ = FiniteContinuousZernike(N, points);
Ψ = FiniteZernikeBasis(N, points, 0, 0);
P = Legendre()
Q = Jacobi(1,1)


λ₀, λ₁ = 1e-2, 5e1
# λ₀, λ₁ = 1.0, 1.0
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

rhs(r,θ,z) =  -Δuₑ(r,θ,z)/k + λ(r)*uₑ(r,θ,z)

# RHS in Cartesian coordinates
function rhs_xyz(xy,z)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs(r,θ,z)
end

𝐳 = ClassicalOrthogonalPolynomials.grid(P, N)
PP = plan_transform(P, (N, sum(1:N)), 1)
X = [zeros(sum(1:N), N) for i in 1:Nₕ]
for k in 1:Nₕ
    for (i, z) in zip(1:lastindex(𝐳), 𝐳)
        rhs_Z(xy) = rhs_xyz(xy, z)
        X[1][:,i], X[2][:,i] =  list_2_modaltrav(Ψ, Ψ \ rhs_Z.(axes(Ψ,1)))
    end
end

𝐳p = [1.0;𝐳;-1.0]
# Find coefficient expansion of tensor-product
Fs = [(PP * X[k]')' for k in 1:Nₕ]
FsP = [Fs[k] * P[𝐳p, 1:N]' for k in 1:Nₕ]
zFsP = []
for i in 1:lastindex(𝐳p)
    append!(zFsP, [modaltrav_2_list(Ψ, [FsP[k][:,i] for k in 1:Nₕ])])
end

errs = []
vals_f = []
rs = []
θs = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(Ψ, zFsP[i], N=200)
    rhs_Z(xy) = rhs_xyz(xy, z)
    vals_, err = inf_error(Ψ, θs, rs, vals, rhs_Z) # Check inf-norm errors on the grid
    append!(errs, [err])
    append!(vals_f, [vals])
end
maximum(errs)


write_adi_vals(𝐳p, rs, θs, vals_f)

zval = round(𝐳p[35], digits=2)
plot(Ψ, θs, rs, vals_f[35], ttl=L"f(x,y,%$zval)")
PyPlot.savefig("adi-rhs-2dslice.png", dpi=500)
iθ =25; slice_plot(iθ, θs, rs, vals_f[35], points, ylabel=L"$f(x,y,%$zval)$")
Plots.savefig("adi-rhs-1dslice.pdf")

wQ = Weighted(Q)
D = Derivative(axes(wQ,1))
pA = ((D*wQ)' * (D*wQ)) ./ k
pM = wQ' * wQ


D = Derivative(axes(Φ,1));
A = (D*Φ)' * (D*Φ);
Λ = piecewise_constant_assembly_matrix(Φ, λ);
M = Φ' * Φ;
G = Φ' * Ψ;

Λ = Matrix.(Λ);
M = Matrix.([M...]);
K = Matrix.(A ./ k .+  Λ);
zero_dirichlet_bcs!(Φ, M);
zero_dirichlet_bcs!(Φ, Λ);
zero_dirichlet_bcs!(Φ, K);


as, bs = [], []
Us = []

zFs = []
for i in 1:lastindex(𝐳)
    append!(zFs, [modaltrav_2_list(Ψ, [Fs[k][:,i] for k in 1:Nₕ])])
end

vals_u = []
θs = []
rs = []
errors = []
for n in 5 : 5 : N
    ns = getNs(n)
    
    tpA = pA[1:n, 1:n];
    # Reverse Cholesky
    rtpA = tpA[end:-1:1, end:-1:1]
    pL = cholesky(Symmetric(rtpA)).L
    pL = pL[end:-1:1, end:-1:1]
    @assert pL * pL' ≈ tpA

    # Compute spectrum
    tpM = pM[1:n, 1:n]
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
    nUs = []
    for i in 1:2n-1
        # Compute spectrum
        B = (L[i] \ (L[i] \ Mn[i])') # = L⁻¹ pM L⁻ᵀ
        a, b = eigmin(B), eigmax(B)
        append!(as, [a])
        append!(bs, [b])

        # weak form for RHS
        fp = zeros(size(Gn[i],2), n)
        for j in 1:n fp[:,j] = zFsn[j][i] end
        F_rhs = Matrix(Gn[i])*fp*((wQ'*P)[1:n, 1:n])'  # RHS <f,v>
        # F_rhs[1, :] .= 0;  F_rhs[Nₕ+1, :] .= 0; # annulus
        F_rhs[Nₕ, :] .= 0; #disk

        X = adi(Mn[i], -tpM, Kn[i], tpA, F_rhs, a, b, c, d, tolerance=1e-15)

        U = (pL' \ (pL \ X'))'
        append!(nUs, [U])
        
    end
    print("n = $n, completed ADI loops.")
    append!(Us, [nUs])

    UsP = [nUs[i] * wQ[𝐳p, 1:n]' for i in 1:2n-1]
    zUm = adi_2_list(Φ, wQ, UsP, 𝐳p, N=n)
    errs_u = []
    vals_u = []
    for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
        (θs, rs, vals) = finite_plotvalues(Φ, zUm[i], N=150)
        uₑ_Z(xy) = uₑ_xyz(xy, z)
        vals_, err = inf_error(Φ, θs, rs, vals, uₑ_Z) # Check inf-norm errors on the grid
        append!(errs_u, [err])
        append!(vals_u, [vals])
    end
    append!(errors, [maximum(errs_u)])
    print("   Completed ℓ-∞ error.\n")
end
writedlm("errors-adi.log", errors)



zval = round(𝐳p[35], digits=2)
plot(Φ, θs, rs, vals_u[35], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("adi-sol-2dslice.png", dpi=500)
iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
Plots.savefig("adi-sol-1dslice.pdf")

write_adi_vals(𝐳p, rs, θs, vals_u)
# PyPlot.savefig("soln-slice-z--0.18-plane-wave.png")
# iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
# Plots.plot!([rs[1] rs[2]], [uₑ.(rs[1],θs[1][iθ],𝐳p[35]) uₑ.(rs[2],θs[2][iθ],𝐳p[35])])


errors = readdlm("errors-adi.log")

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
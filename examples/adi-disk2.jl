using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using SparseDiskFEM, Plots, PyPlot # plotting routines
import ForwardDiff: derivative
using DelimitedFiles

ρ = 0.5
N = 60
points = [0;ρ;1]
Nₕ = length(points) - 1
@time Φ = FiniteContinuousZernike(N, points);
Ψ = FiniteZernikeBasis(N, points, 0, 0);
P = Legendre()
Q = Jacobi(1,1)


λ(r) = r ≤ ρ ? 1/2 : r^2
rhs(r,θ,z) = r ≤ 1/2 ? 2*cos(20*r*sin(θ)) * cos(10*z) : cos(10*r*cos(θ)) * cos(10*z)
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
PyPlot.savefig("adi-rhs-2dslice2.png", dpi=500)
iθ =25; slice_plot(iθ, θs, rs, vals_f[35], points, ylabel=L"$f(x,y,%$zval)$")
Plots.savefig("adi-rhs-1dslice2.pdf")

wQ = Weighted(Q)
D = Derivative(axes(wQ,1))
pA = ((D*wQ)' * (D*wQ)) ./ k
pM = wQ' * wQ


D = Derivative(axes(Φ,1));
A = (D*Φ)' * (D*Φ);
Λ = Φ' * (λ.(axes(Φ,1)).* Φ);
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

tpA = pA[1:N, 1:N];
# Reverse Cholesky
rtpA = tpA[end:-1:1, end:-1:1]
pL = cholesky(Symmetric(rtpA)).L
pL = pL[end:-1:1, end:-1:1]
@assert pL * pL' ≈ tpA

# Compute spectrum
tpM = pM[1:N, 1:N]
B = -(pL \ (pL \ tpM)') # = L⁻¹ pM L⁻ᵀ
c, d = eigmin(B), eigmax(B)

# Reverse Cholesky
rK = [Ks[end:-1:1, end:-1:1] for Ks in K];
Lb = [cholesky(Symmetric(rKs)).L for rKs in rK];
L = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
@assert norm(K .- L.*transpose.(L), Inf) < 1e-10

zFsn = [[zFs[iz][j][1:N*Nₕ] for (N,j) in zip(ns,1:lastindex(zFs[iz]))] for iz in 1:lastindex(zFs)]
Us = []
for i in 1:2N-1
    # Compute spectrum
    B = (L[i] \ (L[i] \ M[i])') # = L⁻¹ pM L⁻ᵀ
    a, b = eigmin(B), eigmax(B)
    append!(as, [a])
    append!(bs, [b])

    # weak form for RHS
    fp = zeros(size(G[i],2), N)
    for j in 1:N fp[:,j] = zFsn[j][i] end
    F_rhs = Matrix(G[i])*fp*((wQ'*P)[1:N, 1:N])'  # RHS <f,v>
    # F_rhs[1, :] .= 0;  F_rhs[Nₕ+1, :] .= 0; # annulus
    F_rhs[Nₕ, :] .= 0; #disk

    X = adi(M[i], -tpM, K[i], tpA, F_rhs, a, b, c, d, tolerance=1e-15)

    U = (pL' \ (pL \ X'))'
    append!(Us, [U])
    
end

UsP = [Us[i] * wQ[𝐳p, 1:n]' for i in 1:2N-1]
zUm = adi_2_list(Φ, wQ, UsP, 𝐳p)
errs_u = []
vals_u = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(Φ, zUm[i], N=200)
    append!(vals_u, [vals])
end

zval = round(𝐳p[35], digits=2)
plot(Φ, θs, rs, vals_u[35], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("adi-sol-2dslice2.png", dpi=500)
iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
Plots.savefig("adi-sol-1dslice2.pdf")

write_adi_vals(𝐳p, rs, θs, vals_u)
# PyPlot.savefig("soln-slice-z--0.18-plane-wave.png")
# iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
# Plots.plot!([rs[1] rs[2]], [uₑ.(rs[1],θs[1][iθ],𝐳p[35]) uₑ.(rs[2],θs[2][iθ],𝐳p[35])])
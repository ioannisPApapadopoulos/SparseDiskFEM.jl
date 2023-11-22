using RadialPiecewisePolynomials, PiecewiseOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using SparseDiskFEM, Plots, PyPlot # plotting routines
import ForwardDiff: derivative
using DelimitedFiles

ρ = 0.5
N = 60
Nz = 30
points = [0;ρ;1]
Nₕ = length(points) - 1
@time Φ = FiniteContinuousZernike(N, points);
Ψ = FiniteZernikeBasis(N, points, 0, 0);



r = range(-1, 1; length=3)
Nzₕ = length(r)-1
P = ContinuousPolynomial{0}(r)
Q = ContinuousPolynomial{1}(r)
# Q = Jacobi(1,1)


λ(r) = r ≤ ρ ? 1/2 : r^2
# rhs(r,θ,z) = r ≤ 1/2 ? 2*cos(20*r*sin(θ)) * cos(10*z) : cos(10*r*cos(θ)) * cos(10*z)

function rhs(r,θ,z)
    f1 = r ≤ 1/2 ? 2*cos(20*r*sin(θ)) : cos(10*r*cos(θ))
    f2 = z ≤ 0 ? 2*cos(20z) : sin(10z)
    f1*f2
end


# RHS in Cartesian coordinates
function rhs_xyz(xy,z)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs(r,θ,z)
end

𝐳 = ClassicalOrthogonalPolynomials.grid(P, Nzₕ*Nz)
# 𝐳 = reverse(vec(𝐳[:,end:-1:1]))
v𝐳 = vec(𝐳)
X = [zeros(sum(1:N), lastindex(v𝐳)) for i in 1:Nₕ]
# for k in 1:Nₕ
    for (i, z) in zip(1:lastindex(v𝐳), v𝐳)
        rhs_Z(xy) = rhs_xyz(xy, z)
        X[1][:,i], X[2][:,i] =  list_2_modaltrav(Ψ, Ψ \ rhs_Z.(axes(Ψ,1)))
    end
# end

# PP*reshape(X[1]', lastindex(𝐳), Nzₕ, sum(1:N))

# Fs = PP * reshape(repeat(X[1]',Nzₕ), Nz, Nzₕ, sum(1:N))


# Find coefficient expansion of tensor-product
PP = plan_transform(P, (Block(Nz), sum(1:N)), 1)
# Fs = [(PP * reshape(repeat(X[k]',Nzₕ),lastindex(𝐳),Nzₕ,sum(1:N)))' for k in 1:Nₕ]
Fs = [(PP * reshape(X[k]', lastindex(𝐳)÷Nzₕ, Nzₕ, sum(1:N)))' for k in 1:Nₕ]

𝐳p = [-1.0;reverse(vec(𝐳[:,end:-1:1]));1.0]

𝐳p = range(-1,1,300)
FsP = [Fs[k] * P[𝐳p, Block.(1:Nz)]' for k in 1:Nₕ]
zFsP = []
for i in 1:lastindex(𝐳p)
    append!(zFsP, [modaltrav_2_list(Ψ, [FsP[k][:,i] for k in 1:Nₕ])])
end

errs = []
vals_f = []
vals_f_ = []
rs = []
θs = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(Ψ, zFsP[i], N=200)
    rhs_Z(xy) = rhs_xyz(xy, z)
    vals_, err = inf_error(Ψ, θs, rs, vals, rhs_Z) # Check inf-norm errors on the grid
    append!(errs, [err])
    append!(vals_f, [vals])
    append!(vals_f_, [vals_])
end
maximum(errs)


write_adi_vals(𝐳p, rs, θs, vals_f)
zval = round(𝐳p[30], digits=2)
SparseDiskFEM.plot(Ψ, θs, rs, vals_f[30], ttl=L"f(x,y,%$zval)")
PyPlot.savefig("adi-rhs-2dslice2.png", dpi=500)
# iθ =25; slice_plot(iθ, θs, rs, vals_f[35], points, ylabel=L"$f(x,y,%$zval)$")
# Plots.savefig("adi-rhs-1dslice2.pdf")

iθ=25; θval = round(θs[1][iθ], digits=4)
SparseDiskFEM.zplot(iθ, rs, 𝐳p, vals_f, ttl=L"f(x,y,z)", xlabel=L"r", ylabel=L"z", title=L"\theta = %$(θval)")

D = Derivative(axes(Q,1))
pA = ((D*Q)' * (D*Q))
pM = Q' * Q
pG = Matrix((Q'*P)[Block.(1:Nz), Block.(1:Nz)]);

D = Derivative(axes(Φ,1));
A = (D*Φ)' * (D*Φ);
Λ = Φ' * (λ.(axes(Φ,1)).* Φ);
M = Φ' * Φ;
G = Φ' * Ψ;

Λ = Matrix.(Λ);
M = Matrix.([M...]);
K = Matrix.(A .+  Λ);
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

tpA = Matrix(pA[Block.(1:Nz), Block.(1:Nz)]);
# Reverse Cholesky
tpA[:,1] .= 0; tpA[1,:] .= 0; tpA[1,1] = 1.0;
tpA[:,Nzₕ+1] .= 0; tpA[Nzₕ+1,:] .= 0; tpA[Nzₕ+1,Nzₕ+1] = 1.0;

rtpA = tpA[end:-1:1, end:-1:1]
pL = cholesky(Symmetric(rtpA)).L
pL = pL[end:-1:1, end:-1:1]
@assert pL * pL' ≈ tpA

# Compute spectrum
tpM = Matrix(pM[Block.(1:Nz), Block.(1:Nz)])
B = -(pL \ (pL \ tpM)') # = L⁻¹ pM L⁻ᵀ
c, d = eigmin(B), eigmax(B)

# Reverse Cholesky
rK = [Ks[end:-1:1, end:-1:1] for Ks in K];
Lb = [cholesky(Symmetric(rKs)).L for rKs in rK];
L = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
@assert norm(K .- L.*transpose.(L), Inf) < 1e-10

ns = getNs(N)
zFsn = [[zFs[iz][j][1:N*Nₕ] for (N,j) in zip(ns,1:lastindex(zFs[iz]))] for iz in 1:lastindex(zFs)]
Us = []
for i in 1:2N-1
# i = 1
    # Compute spectrum
    B = (L[i] \ (L[i] \ M[i])') # = L⁻¹ pM L⁻ᵀ
    a, b = eigmin(B), eigmax(B)
    append!(as, [a])
    append!(bs, [b])

    # weak form for RHS
    fp = zeros(size(G[i],2), size(pG,2))
    for j in 1:Nz fp[:,j] = zFsn[j][i] end
    F_rhs = Matrix(G[i])*fp*pG'  # RHS <f,v>
    # F_rhs[1, :] .= 0;  F_rhs[Nₕ+1, :] .= 0; # annulus
    F_rhs[Nₕ, :] .= 0; #disk
    F_rhs[:, 1] .=0; F_rhs[:, Nzₕ+1] .= 0;

    X = adi(M[i], -tpM, K[i], tpA, F_rhs, a, b, c, d, tolerance=1e-15)

    U = (pL' \ (pL \ X'))'
    append!(Us, [U])
    
end

UsP = [Us[i] * Q[𝐳p, Block.(1:Nz)]' for i in 1:2N-1]
zUm = adi_2_list(Φ, Q, UsP, 𝐳p)
errs_u = []
vals_u = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(Φ, zUm[i], N=200)
    append!(vals_u, [vals])
end

zval = round(𝐳p[50], digits=2)
SparseDiskFEM.plot(Φ, θs, rs, vals_u[35], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("adi-sol-2dslice2.png", dpi=500)
iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
Plots.savefig("adi-sol-1dslice2.pdf")

write_adi_vals(𝐳p, rs, θs, vals_u)
# PyPlot.savefig("soln-slice-z--0.18-plane-wave.png")
# iθ =25; slice_plot(iθ, θs, rs, vals_u[35], points, ylabel=L"$u(x,y,%$zval)$")
# Plots.plot!([rs[1] rs[2]], [uₑ.(rs[1],θs[1][iθ],𝐳p[35]) uₑ.(rs[2],θs[2][iθ],𝐳p[35])])
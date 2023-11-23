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

λ(r) = r ≤ ρ ? 1/2 : r^2
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


# Expand in disk basis for each point on the interval grid
𝐳 = ClassicalOrthogonalPolynomials.grid(P, Nzₕ*Nz)
v𝐳 = vec(𝐳)
X = disk_tensor_transform(Ψ, v𝐳, rhs_xyz, N);

# Expand in interval basis
PP = plan_transform(P, (Block(Nz), sum(1:N)), 1)
# Fs is the list of matrices of expansion coefficients for the RHS.
Fs = [(PP * reshape(X[k]', lastindex(𝐳)÷Nzₕ, Nzₕ, sum(1:N)))' for k in 1:Nₕ];


# Evaluation points
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
Λ = Φ' * (λ.(axes(Φ,1)).* Φ);
M = Φ' * Φ;
G = Φ' * Ψ;

Λ = Matrix.(Λ);
M = Matrix.([M...]);
K = Matrix.(A .+  Λ);
zero_dirichlet_bcs!(Φ, M);
zero_dirichlet_bcs!(Φ, Λ);
zero_dirichlet_bcs!(Φ, K);

zFs = []
for i in 1:lastindex(𝐳)
    append!(zFs, [modaltrav_2_list(Ψ, [Fs[k][:,i] for k in 1:Nₕ])])
end


ns = getNs(N)

tpA = Matrix(pA[Block.(1:N), Block.(1:N)]);
# Reverse Cholesky
tpA[:,1] .= 0; tpA[1,:] .= 0; tpA[1,1] = 1.0;
tpA[:,Nzₕ+1] .= 0; tpA[Nzₕ+1,:] .= 0; tpA[Nzₕ+1,Nzₕ+1] = 1.0;

rtpA = tpA[end:-1:1, end:-1:1]
pL = cholesky(Symmetric(rtpA)).L
pL = pL[end:-1:1, end:-1:1]
@assert pL * pL' ≈ tpA

pG = Matrix(pG[Block.(1:N), Block.(1:N)])

# Compute spectrum
tpM = Matrix(pM[Block.(1:N), Block.(1:N)])
tpM[:,1] .= 0; tpM[1,:] .= 0; tpM[1,1] = 1.0;
tpM[:,Nzₕ+1] .= 0; tpM[Nzₕ+1,:] .= 0; tpM[Nzₕ+1,Nzₕ+1] = 1.0;

B = -(pL \ (pL \ tpM)') # = L⁻¹ pM L⁻ᵀ
c, d = eigmin(B), eigmax(B)

# Reverse Cholesky
rK = [Ks[end:-1:1, end:-1:1] for Ks in K];
Lb = [cholesky(Symmetric(rKs)).L for rKs in rK];
L = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
@assert norm(K .- L.*transpose.(L), Inf) < 1e-10


Us, Js = AbstractMatrix{Float64}[], []
for i in 1:2N-1
    # Compute spectrum
    B = (L[i] \ (L[i] \ M[i])') # = L⁻¹ pM L⁻ᵀ
    a, b = eigmin(B), eigmax(B)

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    append!(Js, [Int(ceil(log(16γ)*log(4/1e-15)/π^2))])

    # weak form for RHS
    fp = zeros(size(G[i],2), size(pG,2))
    for j in 1:n fp[:,j] = zFs[j][i] end
    F_rhs = Matrix(G[i])*fp*pG'  # RHS <f,v>
    F_rhs[Nₕ, :] .= 0; # disk bcs
    F_rhs[:, 1] .=0; F_rhs[:, Nzₕ+1] .= 0; # interval bcs

    X = adi(Mn[i], -tpM, Kn[i], tpA, F_rhs, a, b, c, d, tolerance=1e-15)

    U = (pL' \ (pL \ X'))'
    append!(Us, [U])
    
end
vals_u, rs, θs = synthesis_transform(Φ, Q, Us, 𝐳p, N, N)


#### Plotting

### RHS
# Plotting routines
iz= 30
zval = round(𝐳p[iz], digits=2)

# Disk slice
SparseDiskFEM.plot(Ψ, θs, rs, vals_f[iz], ttl=L"f(x,y,%$zval)")
PyPlot.savefig("adi-rhs-2dslice-alternate.png", dpi=500)
# z slice
θval = round(θs[1][iz], digits=4)
SparseDiskFEM.zplot(iz, rs, 𝐳p, vals_f, ttl=L"f(x,y,z)", xlabel=L"r", ylabel=L"z", title=L"\theta = %$(θval)")
PyPlot.savefig("adi-rhs-zslice-alternate.png", dpi=500)
# save for MATLAB 3D Plot.
write_adi_vals(𝐳p, rs, θs, vals_f)
# 1D slice
# SparseDiskFEM.slice_plot(iz, θs, rs, vals_f[iz], points, ylabel=L"$f(x,y,%$zval)$")
# Plots.savefig("adi-rhs-1dslice2.pdf")

## Solution
SparseDiskFEM.plot(Φ, θs, rs, vals_u[iz], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("adi-sol-2dslice-alternate.png", dpi=500)
θval = round(θs[1][iz], digits=4)
SparseDiskFEM.zplot(iz, rs, 𝐳p, vals_u, ttl=L"u(x,y,z)", xlabel=L"r", ylabel=L"z", title=L"\theta = %$(θval)")
PyPlot.savefig("adi-sol-zslice-alternate.png", dpi=500)
write_adi_vals(𝐳p, rs, θs, vals_u)
# SparseDiskFEM.slice_plot(iz, θs, rs, vals_u[iz], points, ylabel=L"$u(x,y,%$zval)$")
# Plots.savefig("adi-sol-1dslice.pdf")
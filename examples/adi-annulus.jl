using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using SparseDiskFEM # plotting routines

P = Legendre()
Q = Jacobi(1,1)
ρ = 0.6
n, points, k = 60, [0.2;ρ;1.0], 1
K = length(points) - 1
@time F = FiniteContinuousZernike(n, points);
Z = FiniteZernikeBasis(n, points, 0, 0);

λ(r) = r ≤ ρ ? 1.0 : 2.0
function rhs(xy, z)
    x,y = first(xy), last(xy)
    r = sqrt(x^2 + y^2)
    if r ≤ ρ
        return 2*cos(20*y) * cos(10*z)
    else
        return cos(10*x) * cos(10*z)
    end
end

𝐳 = ClassicalOrthogonalPolynomials.grid(P, n)
PP = plan_transform(P, (n, sum(1:n)), 1)
X = [zeros(sum(1:n), n) for i in 1:K]
for k in 1:K
    for (i, z) in zip(1:lastindex(𝐳), 𝐳)
        rhs_Z(xy) = rhs(xy, z)
        X[1][:,i], X[2][:,i] =  list_2_modaltrav(Z, Z \ rhs_Z.(axes(Z,1)))
    end
end

𝐳p = [1.0;𝐳;-1.0]
# Find coefficient expansion of tensor-product
Fs = [(PP * X[k]')' for k in 1:K]
FsP = [Fs[k] * P[𝐳p, 1:n]' for k in 1:K]
zFsP = []
for i in 1:lastindex(𝐳p)
    append!(zFsP, [modaltrav_2_list(Z, [FsP[k][:,i] for k in 1:K])])
end

errs = []
valss = []
rs = []
θs = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(Z, zFsP[i], N=150)
    rhs_Z(xy) = rhs(xy, z)
    vals_, err = inf_error(Z, θs, rs, vals, rhs_Z) # Check inf-norm errors on the grid
    append!(errs, [err])
    append!(valss, [vals])
end
errs

valrθz = [zeros(length(rs[k]), length(θs[k]), length(𝐳p)) for k in 1:K]
for k in 1:K
    for i in 1:lastindex(𝐳p)
        valrθz[k][:,:,i] = valss[i][k]
    end
end

zval = round(𝐳p[35], digits=2)
plot(Z, θs, rs, valss[35], ttl=L"f(x,y,%$zval)")
PyPlot.savefig("rhs-slice-z--0.18-plane-wave.png")
iθ =25; slice_plot(iθ, θs, rs, valss[35], points, ylabel=L"$f(x,y,%$zval)$")

path="src/plotting/"
writedlm(path*"z.log",𝐳p)
writedlm(path*"r1.log", rs[1])
writedlm(path*"r2.log", rs[2])
writedlm(path*"theta1.log", mod2pi.(θs[1]))
writedlm(path*"theta2.log", mod2pi.(θs[2]))
writedlm(path*"vals1.log", valrθz[1]) 
writedlm(path*"vals2.log", valrθz[2]) 

wQ = Weighted(Q)
D = Derivative(axes(wQ,1))
tΔ = (D*wQ)' * (D*wQ)
tM = wQ' * wQ

# Truncated Laplacian + Dirichlet bcs
tpΔ = Matrix(tΔ[1:n, 1:n]);

# Truncated mass + Dirichlet bcs
tpM = Matrix(tM[1:n, 1:n]);

## Helmholtz addition
tpΔ = tpΔ + k^2/2*tpM

# Reverse Cholesky
rtpΔ = tpΔ[end:-1:1, end:-1:1]
@time tL = cholesky(Symmetric(rtpΔ)).L
tL = tL[end:-1:1, end:-1:1]
# tL * tL' ≈ tpΔ

# Compute spectrum
A = -(tL \ (tL \ tpM)') # = L⁻¹ pM L⁻ᵀ
@time c, d = eigmin(A), eigmax(A)

D = Derivative(axes(F,1));
Δ = (D*F)' * (D*F);
# M = F' * F;
M = piecewise_constant_assembly_matrix(F, λ);
Y = F' * Z;

# Ua = zeros(length(𝐱𝐲vec), length(𝐳))
as, bs = [], []
Us = []
# for (N, m, j) in zip(Ns, ms, js)

zFs = []
for i in 1:lastindex(𝐳)
    append!(zFs, [modaltrav_2_list(Z, [Fs[k][:,i] for k in 1:K])])
end

for i in 1:2n-1
    # N = Ns[1]; m = ms[1]; j = js[1]
    # Truncated Laplacian + Dirichlet bcs
    pΔ = Matrix(Δ[i]);
    pΔ[:,1] .=0; pΔ[1,:] .=0; pΔ[1,1] = 1.;
    pΔ[:,K+1] .=0; pΔ[K+1,:] .=0; pΔ[K+1,K+1] = 1.;

    # Truncated mass + Dirichlet bcs
    pM = Matrix(M[i]);
    pM[:,1] .=0; pM[1,:] .=0; pM[1,1] = 1.;
    pM[:,K+1] .=0; pM[K+1,:] .=0; pM[K+1,K+1] = 1.;

    ## Helmholtz addition
    pΔ = pΔ + k^2/2*pM

    # Reverse Cholesky
    rpΔ = pΔ[end:-1:1, end:-1:1]
    @time L = cholesky(Symmetric(rpΔ)).L
    L = L[end:-1:1, end:-1:1]
    # L * L' ≈ pΔ

    # Compute spectrum
    A = (L \ (L \ pM)') # = L⁻¹ pM L⁻ᵀ
    a, b = eigmin(A), eigmax(A)
    append!(as, [a])
    append!(bs, [b])

    # weak form for RHS
    fp = zeros(size(Y[i],2), n)
    for j in 1:n fp[:,j] = zFs[j][i] end
    F_rhs = Matrix(Y[i])*fp*((wQ'*P)[1:n, 1:n])'  # RHS <f,v>
    F_rhs[1, :] .= 0;  F_rhs[K+1, :] .= 0; 

    tol = 1e-15 # ADI tolerance
    @time X = adi(pM, -tpM, pΔ, tpΔ, F_rhs, a, b, c, d, tolerance=tol)

    # X = UΔ
    U = (tL' \ (tL \ X'))'
    append!(Us, [U])
    # W = zeros(length(𝐳), n)
    # for (z, i) in zip(𝐳, 1:lastindex(𝐳))
    #     W[i,1:n] = Q[z, 1:n]
    # end  
    print("i = $i\n")

end

UsP = [Us[i] * wQ[𝐳p, 1:n]' for i in 1:2n-1]
zUm = adi_2_list(F, wQ, UsP, 𝐳p)
errs_u = []
valss_u = []
θs = []
rs = []
for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
    (θs, rs, vals) = finite_plotvalues(F, zUm[i], N=150)
    # u_exact_Z(xy) = u_exact(xy, z)
    # vals_, err = inf_error(F, θs, rs, vals, u_exact_Z) # Check inf-norm errors on the grid
    # append!(errs_u, [err])
    append!(valss_u, [vals])
end
errs_u
valrθz = [zeros(length(rs[k]), length(θs[k]), length(𝐳p)) for k in 1:K]
for k in 1:K
    for i in 1:lastindex(𝐳p)
        valrθz[k][:,:,i] = valss_u[i][k]
    end
end

zval = round(𝐳p[35], digits=2)
plot(F, θs, rs, valss_u[35], ttl=L"u(x,y,%$zval)")
PyPlot.savefig("soln-slice-z--0.18-plane-wave.png")
iθ =25; slice_plot(iθ, θs, rs, valss_u[35], points, ylabel=L"$u(x,y,%$zval)$")


# function cylinder_plot_save(xy::Matrix{<:RadialCoordinate}, z::AbstractArray, vals::AbstractMatrix, path="src/plotting/")
path="src/plotting/"
    writedlm(path*"z.log",𝐳p)
    writedlm(path*"r1.log", rs[1])
    writedlm(path*"r2.log", rs[2])
    # writedlm(path*"r3.log", rs[3])
    # writedlm(path*"r4.log", rs[4])
    writedlm(path*"theta1.log", mod2pi.(θs[1]))
    writedlm(path*"theta2.log", mod2pi.(θs[2]))
    # writedlm(path*"theta3.log", mod2pi.(θs[3]))
    # writedlm(path*"theta4.log", mod2pi.(θs[4]))
    writedlm(path*"vals1.log", valrθz[1]) 
    writedlm(path*"vals2.log", valrθz[2]) 
    # writedlm(path*"vals3.log", valrθz[3]) 
    # writedlm(path*"vals4.log", valrθz[4]) 
# end

# cylinder_plot_save(𝐱𝐲, 𝐳, Ua)

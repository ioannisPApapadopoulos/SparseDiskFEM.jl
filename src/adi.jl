# These 4 routines from ADI were lifted from Kars' M4R repo.
function mobius(z, a, b, c, d, α)
    t₁ = a*(-α*b + b + α*c + c) - 2b*c
    t₂ = a*(α*(b+c) - b + c) - 2α*b*c
    t₃ = 2a - (α+1)*b + (α-1)*c
    t₄ = -α*(-2a+b+c) - b + c

    (t₁*z + t₂)/(t₃*z + t₄)
end

# elliptick(z) = convert(eltype(α),π)/2*HypergeometricFunctions._₂F₁(one(α)/2,one(α)/2,1, z)
function ADI_shifts(J, a, b, c, d, tol=1e-15)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    α = -1 + 2γ + 2√Complex(γ^2 - γ)
    α = Real(α)

    # K = elliptick(1-1/big(α)^2)
    if α < 1e7
        K = Elliptic.K(1-1/α^2)
        dn = Elliptic.Jacobi.dn.((2*(0:J-1) .+ 1)*K/(2J), 1-1/α^2)
    else
        K = 2log(2)+log(α) + (-1+2log(2)+log(α))/α^2/4
        m1 = 1/α^2
        u = (1/2:J-1/2) * K/J
        dn = @. sech(u) + m1/4 * (sinh(u)cosh(u) + u) * tanh(u) * sech(u)
    end

    [mobius(-α*i, a, b, c, d, α) for i = dn], [mobius(α*i, a, b, c, d, α) for i = dn]
end

function adi(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}, F::AbstractMatrix{T}, a::T, b::T, c::T, d::T; tolerance=1e-15) where T
    X = zeros((size(A,1), size(B,1)))

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tolerance)/π^2))

    p, q = ADI_shifts(J, a, b, c, d, tolerance)

    for j = 1:J
        X = ((A/p[j] - C)*X - F/p[j])/(D - B/p[j])
        X = (C - A/q[j])\(X*(B/q[j] - D) - F/q[j])
    end

    X
end
adi(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, F::AbstractMatrix{T}, a::T, b::T, c::T, d::T; tolerance=1e-15) where T =
    adi(A,B,C,C,F,a,b,c,d,tolerance=tolerance)



### This helper functions takes in the coordinates and vals and saves the relevant logs
### to be used with the MATLAB script for plotting solutions on cylinders
function cylinder_plot_save(xy::Matrix{<:RadialCoordinate}, z::AbstractArray, vals::AbstractMatrix, path="src/plotting/")
    writedlm(path*"z.log", z)
    r = [xy[i,1].r for i in 1:size(xy,1)]
    writedlm(path*"r.log", r)
    θ = [xy[1,j].θ for j in 1:size(xy,2)]
    writedlm(path*"theta.log", θ)
    
    writedlm(path*"vals.log", reshape(vals, length(r), length(θ), length(z))) 
end

function write_adi_vals(𝐳p, rs, θs, vals; path="src/plotting/")
    !isdir(path) && mkdir(path)
    Nₕ = length(rs)
    valrθz = [zeros(length(rs[k]), length(θs[k]), length(𝐳p)) for k in 1:Nₕ]
    for k in 1:Nₕ
        for i in 1:lastindex(𝐳p)
            valrθz[k][:,:,i] = vals[i][k]
        end
    end
    writedlm(path*"z.log",𝐳p)
    for i in 1:lastindex(rs)
        writedlm(path*"r$i.log", rs[i])
        writedlm(path*"theta$i.log", mod2pi.(θs[i]))
        writedlm(path*"vals$i.log", valrθz[i]) 
    end
end


function modaltrav_2_list(Φ::ContinuousZernike{T}, u::AbstractArray{Vector{T}}) where T
    N, points = Φ.N, Φ.points
    K = length(points) - 1
    Ns, _, _ = _getMs_ms_js(N)

    cs = []
    u = ModalTrav.(u)
    for i in 1:2N-1
        v = zeros(T, Ns[i]-1, K)
        for k in 1:K
            v[:, k] = u[k].matrix[1:Ns[i]-1,i]
        end
        append!(cs, [pad(vec(v'), blockedrange(Fill(K, Ns[i]-1)))])
    end
    return cs
end

function adi_2_modaltrav(Φ::ContinuousZernike{T}, wQ::Weighted{<:Any, <:ClassicalOrthogonalPolynomials.Jacobi}, Us::AbstractArray, z::AbstractArray{T}) where T
    N, points = Φ.N, Φ.points
    K = length(points) - 1
    Ns, _, _ = _getMs_ms_js(N)

    Y =  [zeros(T, sum(1:N), length(z)) for k in 1:K]
    for zi in 1:lastindex(z)
        X = [zeros(T, Ns[1], 2N-1) for k in 1:K]
        for k in 1:K
            for n in 1:2N-1
                us = Us[n][k:K:end, zi]
                X[k][1:lastindex(us), n] = us
            end
            Y[k][:,zi] = ModalTrav(X[k])
        end
    end
    return Y
end


function axes_Φ(N, points)
    points[1] ≈ 0 && return blockedrange(Vcat(length(points)-1, Fill(length(points) - 1, N-2)))
    blockedrange(Vcat(length(points), Fill(length(points) - 1, N-2)))
end

function _adi_2_list(Φ::ContinuousZernike{T}, Us::AbstractArray, z::AbstractArray{T}; N=0) where T
    points = Φ.points
    K = length(points) - 1

    N = N==0 ? Int((length(Us)+1)/2) : N
    @assert N ≤ Φ.N

    Ns = getNs(N)

    Y = []
    Fs = Φ.Fs
    for zi in 1:lastindex(z)
        cs = []
        for n in 1:2N-1

            # Nn = points[1] ≈ 0 ? Ns[n]-1 : Ns[n]
            # v = zeros(T, Nn, K)
            # for k in 1:K
            #     v[:, k] = Us[n][k:K:end, zi]
            # end
            # append!(cs, [pad(vec(v'), blockedrange(Fill(K, Nn)))])

            append!(cs, [pad(Us[n][:,zi], axes_Φ(Ns[n], points))])
        end
        append!(Y, [cs])
    end
    return Y



    bubbles = zeros(T, N-2, K)
    if first(points) ≈ 0
        if K == 1
            hats = [fs[1][1]]
        else
            hats = vcat([fs[i][1] for i in 2:K-1], fs[end][1:2])
        end
        bubbles[:,1] = fs[1][2:N-1]
        for i in 2:K bubbles[:,i] = fs[i][3:N] end
    else
        hats = vcat([fs[i][1] for i in 1:K-1], fs[end][1:2])
        for i in 1:K bubbles[:,i] = fs[i][3:N] end
    end

    pad(append!(hats, vec(bubbles')), axes(F,2))
end

adi_2_list(Φ::ContinuousZernike{T}, wQ::Weighted{<:Any, <:ClassicalOrthogonalPolynomials.Jacobi}, Us::AbstractArray, z::AbstractArray{T};N=0) where T = _adi_2_list(Φ, Us, z, N=N)
adi_2_list(Φ::ContinuousZernike{T}, Q::ContinuousPolynomial, Us::AbstractArray, z::AbstractArray{T};N=0) where T = _adi_2_list(Φ, Us, z, N=N)



## Coefficient storage conversion
function list_2_modaltrav(Ψ::ZernikeBasis{T}, u::AbstractArray) where T
    N, points = Ψ.N, Ψ.points
    K = length(points) - 1
    U = [zeros(T, N ÷ 2, 2N-1) for i in 1:K]
    Ns, _, _ = _getMs_ms_js(N)
    for k in 1:K
        for i in 1:lastindex(Ns)
            U[k][1:Ns[i], i] = u[i][k:K:end]
        end
    end
    return ModalTrav.(U)
end

function modaltrav_2_list(Ψ::ZernikeBasis{T}, u::AbstractArray{Vector{T}}) where T
    N, points = Ψ.N, Ψ.points
    K = length(points) - 1
    Ns, _, _ = _getMs_ms_js(N)

    cs = []
    u = ModalTrav.(u)
    for i in 1:2N-1
        v = zeros(T, Ns[i], K)
        for k in 1:K
            v[:, k] = u[k].matrix[1:Ns[i],i]
        end
        append!(cs, [pad(vec(v'), blockedrange(Fill(K, Ns[i])))])
    end
    return cs
end

function adi_2_modaltrav(Ψ::ZernikeBasis{T}, P::Legendre{T}, Us::AbstractArray, z::AbstractArray{T}) where T
    N, points = Ψ.N, Ψ.points
    K = length(points) - 1
    Ns, _, _ = _getMs_ms_js(N)

    Y =  [zeros(T, sum(1:N), length(z)) for k in 1:K]
    for i in 1:lastindex(z)
        X = [zeros(T, Ns[1], 2N-1) for k in 1:K]
        for k in 1:K
            for n in 1:2N-1
                us = Us[n][k:K:end, i]
                X[k][1:lastindex(us), n] = us
            end
            Y[k][:,i] = ModalTrav(X[k])
        end
    end
    return Y
end

function disk_tensor_transform(Ψ::ZernikeBasis{T}, v𝐳::AbstractVector{T}, rhs_xyz::Function, N::Int) where T
    Nₕ = length(Ψ.points) - 1
    @assert Nₕ == 2
    X = [zeros(sum(1:N), lastindex(v𝐳)) for i in 1:Nₕ]
    for (i, z) in zip(1:lastindex(v𝐳), v𝐳)
        rhs_Z(xy) = rhs_xyz(xy, z)
        X[1][:,i], X[2][:,i] =  list_2_modaltrav(Ψ, Ψ \ rhs_Z.(axes(Ψ,1)))
    end
    X
end

function _synthesis_error_transform(Ψ, zFsP::AbstractVector, 𝐳p::AbstractVector{T}, rhs_xyz::Function, N::Int) where T
    # Epand out in disk basis via synthesis operators
    errs, vals, rs, θs, vals_errs = [], [], [], [], []
    for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
        (θs, rs, valss) = finite_plotvalues(Ψ, zFsP[i], N=N)
        rhs_Z(xy) = rhs_xyz(xy, z)
        vals_err, err = inf_error(Ψ, θs, rs, valss, rhs_Z) # Check inf-norm errors on the grid
        append!(errs, [err])
        append!(vals, [valss])
        append!(vals_errs, [vals_err])
    end
    vals, rs, θs, vals_errs, errs
end

function synthesis_error_transform(Ψ::ZernikeBasis{T}, P::ContinuousPolynomial{0}, Fs::AbstractVector{<:AbstractMatrix{<:T}}, 𝐳p::AbstractVector{T}, rhs_xyz::Function, N::Int, Nz::Int) where T 
    Nₕ = length(Ψ.points) - 1
    # Expand out in interval basis at points 𝐳p
    FsP = [Fs[k] * P[𝐳p, Block.(1:Nz)]' for k in 1:Nₕ]
    zFsP = []
    for i in 1:lastindex(𝐳p)
        append!(zFsP, [modaltrav_2_list(Ψ, [FsP[k][:,i] for k in 1:Nₕ])])
    end
    _synthesis_error_transform(Ψ, zFsP, 𝐳p, rhs_xyz, 150)
end


function synthesis_error_transform(Φ::ContinuousZernike{T}, Q::ContinuousPolynomial{1}, Us::AbstractVector{<:AbstractMatrix{<:T}}, 𝐳p::AbstractVector{T}, u_xyz::Function, N::Int, Nz::Int) where T 
    UsP = [Us[i] * Q[𝐳p, Block.(1:Nz)]' for i in 1:2N-1]
    zUm = adi_2_list(Φ, Q, UsP, 𝐳p, N=N)
    _synthesis_error_transform(Φ, zUm, 𝐳p, u_xyz, 150)
end


function _synthesis_transform(Ψ, zFsP, 𝐳p::AbstractVector{T}, N::Int) where T
    # Epand out in disk basis via synthesis operators
    vals, rs, θs = [], [], []
    for (i, z) in zip(1:lastindex(𝐳p), 𝐳p)
        (θs, rs, valss) = finite_plotvalues(Ψ, zFsP[i], N=N)
        rhs_Z(xy) = rhs_xyz(xy, z)
        append!(vals, [valss])
    end
    vals, rs, θs
end

function synthesis_transform(Ψ::ZernikeBasis{T}, P::ContinuousPolynomial{0}, Fs::AbstractVector{<:AbstractMatrix{<:T}}, 𝐳p::AbstractVector{T}, N::Int, Nz::Int) where T 
    Nₕ = length(Ψ.points) - 1
    # Expand out in interval basis at points 𝐳p
    FsP = [Fs[k] * P[𝐳p, Block.(1:Nz)]' for k in 1:Nₕ]
    zFsP = []
    for i in 1:lastindex(𝐳p)
        append!(zFsP, [modaltrav_2_list(Ψ, [FsP[k][:,i] for k in 1:Nₕ])])
    end
    _synthesis_transform(Ψ, zFsP, 𝐳p, 150)
end

function synthesis_transform(Φ::ContinuousZernike{T}, Q::ContinuousPolynomial{1}, Us::AbstractVector{<:AbstractMatrix{<:T}}, 𝐳p::AbstractVector{T}, N::Int, Nz::Int) where T 
    UsP = [Us[i] * Q[𝐳p, Block.(1:Nz)]' for i in 1:2N-1]
    zUm = adi_2_list(Φ, Q, UsP, 𝐳p, N=N)
    _synthesis_transform(Φ, zUm, 𝐳p, 150)
end
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


function modaltrav_2_list(Φ::FiniteContinuousZernike{T}, u::AbstractArray{Vector{T}}) where T
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

function adi_2_modaltrav(Φ::FiniteContinuousZernike{T}, wQ::Weighted{<:Any, <:Jacobi}, Us::AbstractArray, z::AbstractArray{T}) where T
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

function _adi_2_list(Φ::FiniteContinuousZernike{T}, Us::AbstractArray, z::AbstractArray{T}; N=0) where T
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

adi_2_list(Φ::FiniteContinuousZernike{T}, wQ::Weighted{<:Any, <:Jacobi}, Us::AbstractArray, z::AbstractArray{T};N=0) where T = _adi_2_list(Φ, Us, z, N=N)
adi_2_list(Φ::FiniteContinuousZernike{T}, Q::ContinuousPolynomial, Us::AbstractArray, z::AbstractArray{T};N=0) where T = _adi_2_list(Φ, Us, z, N=N)



## Coefficient storage conversion
function list_2_modaltrav(Ψ::FiniteZernikeBasis{T}, u::AbstractArray) where T
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

function modaltrav_2_list(Ψ::FiniteZernikeBasis{T}, u::AbstractArray{Vector{T}}) where T
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

function adi_2_modaltrav(Ψ::FiniteZernikeBasis{T}, P::Legendre{T}, Us::AbstractArray, z::AbstractArray{T}) where T
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
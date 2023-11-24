using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlternatingDirectionImplicit
using DelimitedFiles
using Plots

ρ = 0.5
N = 20

as, bs = [], []
Js, γs = [], []
Nhs = 1:20
for Nh in Nhs
    s = ρ^(-1/Nh)
    points = [0; reverse([s^(-j) for j in 0:Nh])]
    # points = [0;ρ;1]
    Nₕ = length(points) - 1
    @time Φ = FiniteContinuousZernike(N, points);
    Ψ = FiniteZernikeBasis(N, points, 0, 0);

    D = Derivative(axes(Φ,1));
    A = (D*Φ)' * (D*Φ);
    # Λ = Φ' * (λ.(axes(Φ,1)).*Φ) #piecewise_constant_assembly_matrix(Φ, λ);

    M = Φ' * Φ;
    # Λ = Matrix.(Λ);
    M = Matrix.([M...]);
    K = Matrix.(A .+  M);
    zero_dirichlet_bcs!(Φ, M);
    # zero_dirichlet_bcs!(Φ, [Λ...]);
    zero_dirichlet_bcs!(Φ, [K...]);
    Λ = M;

    r = range(-1, 1; length=Nh+2)
    Nzₕ = Nh + 1
    # Interval FEM basis
    P = ContinuousPolynomial{0}(r)
    Q = ContinuousPolynomial{1}(r)
    D = Derivative(axes(Q,1))
    pA = ((D*Q)' * (D*Q))
    pM = Q' * Q

    for n in [Nh+1]
        ns = getNs(n)
        
        tpA = Matrix(pA[Block.(1:n), Block.(1:n)]);
        tpA[:,1] .= 0; tpA[1,:] .= 0; tpA[1,1] = 1.0;
        tpA[:,Nzₕ+1] .= 0; tpA[Nzₕ+1,:] .= 0; tpA[Nzₕ+1,Nzₕ+1] = 1.0;
        # Reverse Cholesky
        rtpA = tpA[end:-1:1, end:-1:1]
        pL = cholesky(Symmetric(rtpA)).L
        pL = pL[end:-1:1, end:-1:1]
        @assert pL * pL' ≈ tpA

        # Compute spectrum
        tpM = Matrix(pM[Block.(1:n), Block.(1:n)])
        tpM[:,1] .= 0; tpM[1,:] .= 0; tpM[1,1] = 1.0;
        tpM[:,Nzₕ+1] .= 0; tpM[Nzₕ+1,:] .= 0; tpM[Nzₕ+1,Nzₕ+1] = 1.0;
        B = -(pL \ (pL \ tpM)') # = L⁻¹ pM L⁻ᵀ
        c, d = eigmin(B), eigmax(B)

    
        Kn = [K[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(K))];
        Mn = [M[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(M))];
        Λn = [Λ[j][1:(N-1)*Nₕ, 1:(N-1)*Nₕ] for (N,j) in zip(ns,1:lastindex(Λ))];
        # Reverse Cholesky
        rK = [Ks[end:-1:1, end:-1:1] for Ks in Kn];
        Lb = [cholesky(Symmetric(rKs)).L for rKs in rK];
        L = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
        @assert norm(Kn .- L.*transpose.(L), Inf) < 1e-10

        for i in 1:1
            # Compute spectrum
            B = (L[i] \ (L[i] \ Mn[i])') # = L⁻¹ pM L⁻ᵀ
            a, b = eigmin(B), eigmax(B)
            append!(as, [a])
            append!(bs, [b])
            
            γ = (c-a)*(d-b)/((c-b)*(d-a))
            J = Int(ceil(log(16γ)*log(4/1e-15)/π^2))

            append!(γs, [γ])
            append!(Js, [J])
        end
        print("n = $n, completed ADI loops.")


    end
end


h = (Nhs .+ 1)[1:end-1]

rate = 1 ./ (Nhs .+ 1).^3 * as[1] * (Nhs[1] + 1)^3
Plots.plot(log.(h .* h.^(1/2)), Js)
Plots.plot!(Nhs .+ 1, rate, yaxis=:log10, xaxis=:log10)

Plots.plot(Nhs .+ 1, γs)
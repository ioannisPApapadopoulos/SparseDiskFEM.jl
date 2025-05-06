using SparseDiskFEM, RadialPiecewisePolynomials, SemiclassicalOrthogonalPolynomials
using BlockDiagonals, BandedMatrices, BlockBandedMatrices, LinearAlgebra
using SparseArrays, ExtendableSparse, MatrixFactorizations
using PyPlot, Plots
import SemiclassicalOrthogonalPolynomials: BroadcastVector, Weighted
import RadialPiecewisePolynomials: _mass_m₀, Fill, _getγs


"""
Section 6.4 "High frequency with a non-separable Helmholtz coefficient and 
discontinuous right-hand side".

Domain is Ω = {0.01 ≤ r ≤ 1} and we are solving
    (-Δ - k² x)u(x,y) = f(x,y).
This problem is non-separable, i.e. unlike the other examples the discretization
does not result in a block-diagonal matrix of Fourier modes.

This does not quite fit in the framework of RadialPiecewisePolynomials.jl and hence
we require some work to assemble the FEM discretization.

The operator x is discretized by x-Jacobi matrix leveraging Proposition 4.8 
from https://doi.org/10.1137/23M160846X.

If R is the lowering matrix from the continuous FEM basis Φ to the DG basis Z^(0,0), then

    x ≈ Rᵀ D X R
where D is the diagonal orthogonality matrix of Z and X is the x-Jacobi matrix for 
the DG basis Z^(0,0) as computed in Proposition 4.8.

"""

T = Float64
N = 100 # Discretization degree
k = 80  # frequency
r1, r2 = 0.01, 0.5 # Two cells mesh, inner-radius = 0.01
points = [r1;r2;1.0]

Φ = ContinuousZernike(N, points) 

# Collect Fourier mode components over both cells of lowering matrix R 
# and scaling matrix D for full assembly.
Rs = [[],[]]
Ds = [[],[]]
for i in 1:length(Φ.Fs)
    for j in 1:2
        C = Φ.Fs[i].Cs[j]
        R = C.R
        α, β = convert(T, first(C.points)), convert(T, last(C.points))
        ρ = α / β
        m = C.m
        t = inv(1-ρ^2)
        m₀ = _mass_m₀(C, m, t)
        push!(Rs[j], R)
        push!(Ds[j], Diagonal(Fill(β^2*m₀,size(R,1))))
    end
end

cols = rows = size.(Rs[1],1)  # block sizes

# Interlace the two elements in the lowering matrix, combine 
# the two sides of the middle hat function and enforce continuity.
Rs3 = AbstractMatrix{T}[]
for i in 1:length(Φ.Fs)
    γs = _getγs(points, Φ.Fs[i].m)

    Rs2 = ExtendableSparseMatrix(2*size(Rs[1][i],1),2*size(Rs[1][i],1)-1) 
    Rs2[1:2:end,4:2:end] .= sparse(view(Rs[1][i],1:rows[i],3:cols[i]))
    Rs2[2:2:end,5:2:end] .= sparse(view(Rs[2][i],1:rows[i],3:rows[i]))
    Rs2[1:2:3,1] .= Rs[1][i][1:2,1]
    Rs2[1:2:3,2] .= γs[1] .* Rs[1][i][1:2,2]
    Rs2[2:2:4,2] .= Rs[2][i][1:2,1]
    Rs2[2:2:4,3] .= Rs[2][i][1:2,2]
    push!(Rs3, sparse(Rs2))
end
# Combine the Fourier mode matrices into a block diagonal lowering matrix
BR = ExtendableSparseMatrix(sum(size.(Rs3,1)), sum(size.(Rs3,2)))
BR[1:sum(size(Rs3[1],1)), 1:sum(size(Rs3[1],2))] .= Rs3[1]
for i in 2:length(Φ.Fs)
    BR[1+sum(size.(Rs3[1:i-1],1)):sum(size.(Rs3[1:i],1)), 1+sum(size.(Rs3[1:i-1],2)):sum(size.(Rs3[1:i],2))] .= Rs3[i]
end

# Interlace scaling matrix over the two cells and combine
# Fourier modes
BD = Diagonal.([BlockDiagonal(sparse.(Ds[j])) for j in 1:2])
BD2 = ExtendableSparseMatrix(2*size(BD[1],1), 2*size(BD[1],2))
# BD2 = Diagonal(Zeros(2*size(BD[1],1), 2*size(BD[1],2)))
BD2[1:2:end,1:2:end] .= BD[1]
BD2[2:2:end,2:2:end] .= BD[2]
BD = sparse(BD2)

# Now we compute the x-Jacobi matrix X for the DG space Z 
# by leveraging Proposition 4.8 in https://doi.org/10.1137/23M160846X

# Compute the necessary SemiclassicalJacobi lowering matrices
t = [inv(1-(r1/r2)^2), inv(1-r2^2)]
Q = [SemiclassicalJacobi.(t[j],0,0,1:N) .\ SemiclassicalJacobi.(t[j],0,0,0:N-1) for j in 1:2]
Qt = [BroadcastVector{AbstractMatrix{T}}((A,B) -> Weighted(A) \ Weighted(B), SemiclassicalJacobi.(t[j],0,0,0:N-1), SemiclassicalJacobi.(t[j],0,0,1:N)) for j in 1:2]
Qs = [[Q[1][1][1:N,1:N]],[Q[2][1][1:N,1:N]]]
Qts = [[Qt[1][1][1:N,1:N]], [Qt[2][1][1:N,1:N]]]

# Order them correctly
for i in 2:length(Q[1])
    for j in 1:2
        push!(Qs[j], Q[j][i][1:N,1:N])
        push!(Qs[j], Q[j][i][1:N,1:N])
        push!(Qts[j], Qt[j][i][1:N,1:N])
        push!(Qts[j], Qt[j][i][1:N,1:N])
    end
end

# Construct the full x-Jacobi matrix for the DG space Z
BX =  [BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows, cols, (2,2)) for j in 1:2]
for i in 1:2N-3
    for j in 1:2
        sr,sc = size(BX[j][Block(i+2), Block(i)])
        s = i == 1 ? 1 : 2
        β = j == 1  ? r2 : 1.0
        BX[j][Block(i+2), Block(i)] .= β .* (Qs[j][i][1:sr,1:sc] ./ s)
        BX[j][Block(i), Block(i+2)] .= β .* (Qts[j][i][1:sc,1:sr] ./ (2*t[j]))
    end
end

# Interlace the two cells
BX = sparse.(Matrix.(BX))
BX2 = spzeros(2 .* size(BX[1]))
BX2[1:2:end,1:2:end] .= BX[1]
BX2[2:2:end,2:2:end] .= BX[2]
BX = BX2;

# Assemble the x-Jacobi matrix.
Jx = BR' * (BD * (BX * BR))

### A sanity check that we have assembled the correct matrix Jx.
xy = axes(Φ,1)
fc = Φ \ (xy->exp(first(xy))).(xy)
fc = reduce(vcat,fc)
# fc' * Jx * fc should equal ∫∫ x (eˣ)² dxdy
# NIntegrate[NIntegrate[r*r*Cos[y]*(Exp[r*Cos[y]])^2, {r, 0.01, 1.0}], {y, -Pi, Pi}]
@assert fc' * Jx * fc ≈ 2.16439536628396 


### Construct the FEM linear system

# First the stiffness matrix
A = (Matrix.(stiffness_matrix(Φ)))

# Block-diagonal stiffness matrix of all Fourier modes
BA = ExtendableSparseMatrix(sum(size.(A,1)), sum(size.(A,2)))
BA[1:sum(size(A[1],1)), 1:sum(size(A[1],2))] .= A[1]
for i in 2:length(Φ.Fs)
    print("i=$i.\n")
    BA[1+sum(2 .*rows[1:i-1].-1):sum(2 .*rows[1:i].-1), 1+sum(2 .*rows[1:i-1].-1):sum(2 .*rows[1:i].-1)] .= A[i]
end
BA = sparse(BA)

# FEM linear system matrix
K = BA - k^2 .* Jx

# Enforce zero boundary conditions by zeroing all rows & columns
# associated with the far left and right hat functions across all
# Fourier modes
K[1,:] .= 0; K[3,:].= 0;
K[:,1] .= 0; K[:,3].=0;
K[1,1] = 1.0; K[3,3] = 1.0;
for i in 1 : length(rows)-1
    K[1 + sum(2 .*rows[1:i].-1),:] .= 0; K[3 + sum(2 .*rows[1:i].-1),:] .= 0
    K[:, 1 + sum(2 .*rows[1:i].-1)] .= 0; K[:, 3 + sum(2 .*rows[1:i].-1)] .=0
    K[1 + sum(2 .*rows[1:i].-1),1 + sum(2 .*rows[1:i].-1)] = 1.0; K[3 + sum(2 .*rows[1:i].-1),3 + sum(2 .*rows[1:i].-1)] = 1.0
end


# Choice of right-hand side
function f(xy)
    x, y = first(xy), last(xy)
    if x^2 + y^2 ≤ 0.5^2
        return (1 + exp(-12*first(xy)) ) * sin(50*sin(first(xy)))
    else
        return (1 + exp(-6*first(xy))) * sin(50*sin(last(xy)))
    end
end

# Find coefficients of right-hand side in Ψ-basis
Ψ = ZernikeBasis(N, points, 0, 0);
x = axes(Ψ,1)
fz = Ψ \ f.(x);

# Plot right-hand side and check error
(θs, rs, vals) = finite_plotvalues(Ψ, fz, N=700);
vals_, err = inf_error(Ψ, θs, rs, vals, f); err
SparseDiskFEM.plot(Ψ, θs, rs, vals, ttl=L"f(x,y)", ρ=r1, logscale=true) # plot
PyPlot.savefig("non-separable-rhs.png", dpi=500)

# Create Gram matrix and assemble load vector
G = (Φ' * Ψ);  # <v, u>, v ∈ Φ, u ∈ Ψ
Mf =  G .* fz; # <v, f>
zero_dirichlet_bcs!(Φ, Mf);
Mf = reduce(vcat,Mf)

# Solve!
lu_K = MatrixFactorizations.lu(K)
u = lu_K \ Mf

# Split the solution vector into its Fourier modes for plotting
us = [u[1:2*rows[1]-1]]
for i in 1:length(rows)-1
    push!(us, u[(1+sum(2 .*rows[1:i].-1)):(sum(2 .*rows[1:i+1].-1))])
end
# Check the splitting was done correctly
u2 = reduce(vcat,us)
u2==u

# Plot the solution
(θs, rs, vals) = finite_plotvalues(Φ, us, N=700)
SparseDiskFEM.plot(Φ, θs, rs, vals, ttl=L"u(x,y)", ρ=r1) # plot
PyPlot.savefig("non-separable-sol.png", dpi=500)

slice_plot(100, θs, rs, vals, points, ylabel=L"$u(x,y)$")
Plots.savefig("non-separable-sol-slice.pdf")
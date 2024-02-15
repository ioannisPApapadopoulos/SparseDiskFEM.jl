using AnnuliPDEs, ClassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials, MultivariateOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings
using JLD
import AnnuliPDEs: pad, ZernikeITransform
import ClassicalOrthogonalPolynomials: affine

"""

"""

ρ = 0.5
λ(r) = r ≤ ρ ? 80^2 : 90^2

function f(x,y)
    if x^2 + y^2 ≤ ρ^2
        return 2*sin(200*x)
    else
        return sin(100*y)
    end
end

# Scaled RHS for the disk element
function f_scale(x,y)
    f(ρ*x, ρ*y)
end

Nn = 400
λs = [λ(0.8); λ(0.2)]

###
# (2-element) Spectral element Zernike + Chebyshev-Fourier series discretisation
###
T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r = axes(T,1)
D = Derivative(r)

Lₜ = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Z = Zernike(0,0)
Zd = Zernike(0,2)

Δ =   Zd \ (Laplacian(axes(Z,1)) * Z);
L =  Zd \ Z;
Δs, Ls = Δ.ops, L.ops;

xy = axes(Z,1); x,y = first.(xy),last.(xy)

n = 300
# Expand RHS in Zernike polynomials for the disk element
fz = Zd[:, Block.(1:n)] \ f_scale.(x,y)

# Solve by breaking down into solves for each Fourier mode.
# We utilise a tau-method to enforce the boundary conditions
# and continuity.
X, u = chebyshev_fourier_zernike_helmholtz_modal_solve([(T,F), Z], f, fz, ρ, n, [(Lₜ, M, R, D), Δs], [[], Ls], λs)


cells = 11
u_ = pad(u, axes(Z,2))[Block.(1:n)]
FT = ZernikeITransform{Float64}(n, Z.a, Z.b)
val = FT * (-u_) # synthesis - transform to grid
val = [val val[:,1]]
vals = [val]

s = ρ^(-1/cells)
points = [0; reverse([s^(-j) for j in 0:cells])]
ρs = []
for k = 1:length(points)-1
    α, β = points[k], points[k+1]
    append!(ρs, [α / β])
end
Za = ZernikeAnnulus(ρs[2], 0, 0)
function scalegrid(g::Matrix{RadialCoordinate{T}}, α::T, β::T) where T
    ρ = α / β
    rs = x -> affine(ρ.. 1, α.. β)[x.r]
    gs = (x, r) -> RadialCoordinate(SVector(r*cos(x.θ), r*sin(x.θ)))
    r̃ = map(rs, g)
    gs.(g, r̃)
end
function plot_helper(g::Matrix{RadialCoordinate{T}}) where T
    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    r = first.(rθ)[:,1]
    θ = last.(rθ)[1,:]

    θ = [θ; 2π]
    (θ, r, vals)
end
θs, rs = [], []
for k in 2:lastindex(points)-1
    g = scalegrid(AnnuliOrthogonalPolynomials.grid(Za, Block(n)), points[k], points[k+1])
    (θ, r) = plot_helper(g)
    append!(θs, [θ]); append!(rs, [r])
end
 
for k in 1:lastindex(points)-2
    append!(vals, [-(F[θs[k],1:size(X,2)]*(T[rs[k],1:size(X,1)]*X)')'])
end



JLD.save("high-frequency-soln.jld", "vals", vals)

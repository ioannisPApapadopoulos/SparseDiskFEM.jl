using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials
using Plots, LaTeXStrings

"""
Script to plot the slices at θ=0 of the sparse hp-FEM basis function 
on a mesh with the two cells:
    K₀ = {0 ≤ r ≤ 1/2}
    K₁ = {1/2 < r ≤ 1}
"""

disk_bubbles(n,m,r) = (1-r^2) * r^m * normalizedjacobip(n, 1, m, 2r^2-1)
annulus_bubbles(ρ, n, m, r) =  (1-r^2) * (r^2 - ρ^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,1,m))[(r^2 - 1)/(ρ^2 - 1), n+1]
annulus_hat(ρ, m, r) = (r^2-ρ^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),0,1,m))[(r^2 - 1)/(ρ^2 - 1), 1]
function disk_hat(ρ, m, r)
    if abs(r) ≤ ρ
        κ =  (1-ρ^2)* ρ^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,0,m))[0.1, 1] / normalizedjacobip(0, 0, m, 0.0)
        return κ * (r/ρ)^m * normalizedjacobip(0, 0, m, 2(r/ρ)^2-1)
    else
        return (1-r^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,0,m))[(r^2 - 1)/(ρ^2 - 1), 1]
    end
end

xx = range(-0.5,0.5,200)
plot(xx, [disk_bubbles.(n, 0, xx ./ 0.5) for n in 0:0],
    linewidth = 2,
    label=L"B^{K_0}_{0,0,1}(x,y)")

xx = [range(-1,-0.5,100), range(0.5,1,100)]
plot!(xx, [annulus_bubbles.(0.5, 0, 0, x) for x in xx],
    linewidth = 2,
    color=[:red :red],
    label=[L"B^{K_1}_{0,0,1}(x,y)" ""]
)

xx = range(-1,1,200)
plot!(xx, disk_hat.(0.5, 0, xx),
    linewidth = 2,
    linestyle=:dash,
    label=L"H^{K_0, K_1}_{0,1}(x,y)"
)

xx = [range(-1,-0.5,100), range(0.5,1,100)]
plot!(xx, [annulus_hat.(0.5, 0, x) for x in xx],
    label=[L"H^{K_1, \bullet}_{0,1}(x,y)" ""],
    xlabel = L"r",
    ylabel = L"y",
    xlabelfontsize=20, ylabelfontsize=20, legendfontsize=10,
    xtickfontsize=12, ytickfontsize=12,
    linewidth = 2,
    legend=:bottom,
    color=[:orange :orange],
    linestyle=:dash,
    gridlinewidth = 2,
    legend_hfactor = 1.1,
    extra_kwargs=:subplot
)
vline!([-0.5,0.5], label="", color=:black, linewidth=2)
Plots.savefig("bubble-hats-slice.pdf")



xx = range(-0.5,0.5,200)
p = plot(xx, [disk_bubbles.(n, 0, xx ./ 0.5) for n in 0:3],
    xlabel = L"r",
    ylabel = L"y",
    xlabelfontsize=15, ylabelfontsize=15, legendfontsize=8,
    xtickfontsize=10, ytickfontsize=10,
    linewidth = 2,
    gridlinewidth = 2,
    legend=:bottomright,
    xlim=[-0.54,0.54],
    xticks=[-0.5,-0.25,0,0.25,0.5],
    size=(420,400),
    margin=2Plots.mm,
    label=[L"B^{K_0}_{0,0,1}(x,y)" L"B^{K_0}_{2,0,1}(x,y)" L"B^{K_0}_{4,0,1}(x,y)" L"B^{K_0}_{6,0,1}(x,y)"]
    )
vline!([-0.5,0.5], color=:black, linewidth=2, label="")
Plots.savefig("bubble-slice.pdf")
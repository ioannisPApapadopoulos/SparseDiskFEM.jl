using RadialPiecewisePolynomials
using ClassicalOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials
using Plots, LaTeXStrings

"""
Script to plot the slices at θ=0 of the sparse hp-FEM basis function 
on a mesh with the two cells:
    K₀ = {0 ≤ r ≤ 1/2}
    K₁ = {1/2 < r ≤ 1}
"""

disk_bubbles(n,m,r) = (1-r^2) * r^m * normalizedjacobip((n-m)÷2, 1, m, 2r^2-1)
annulus_bubbles(ρ, n, m, r) =  (1-r^2) * (r^2 - ρ^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,1,m))[(r^2 - 1)/(ρ^2 - 1), (n-m)÷2+1]
annulus_hat(ρ, m, r) = (r^2-ρ^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),0,1,m))[(r^2 - 1)/(ρ^2 - 1), 1]
function disk_hat(ρ, m, r)
    if abs(r) ≤ ρ
        κ =  (1-ρ^2)* ρ^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,0,m))[0.1, 1] / normalizedjacobip(0, 0, m, 0.0)
        return κ * (r/ρ)^m * normalizedjacobip(0, 0, m, 2(r/ρ)^2-1)
    else
        return (1-r^2) * r^m * Normalized(SemiclassicalJacobi(inv(1-ρ^2),1,0,m))[(r^2 - 1)/(ρ^2 - 1), 1]
    end
end

for (n, m) in zip([0,1], [0,1])
# n,m=1,1
    xx = range(-0.5,0.5,200)
    Plots.plot(xx, disk_bubbles.(n, m, xx ./ 0.5),
        linewidth = 2,
        label=L"B^{K_0}_{%$n,%$m,1}(x,y)")

    xx = [range(-1,-0.5,100), range(0.5,1,100)]
    Plots.plot!(xx, [annulus_bubbles.(0.5, n, m, x) for x in xx],
        linewidth = 2,
        color=[:red :red],
        label=[L"B^{K_1}_{%$n,%$m,1}(x,y)" ""]
    )

    xx = range(-1,1,200)
    Plots.plot!(xx, disk_hat.(0.5, m, xx),
        linewidth = 2,
        linestyle=:dash,
        label=L"H^{K_0, K_1}_{%$m,1}(x,y)"
    )

    xx = [range(-1,-0.5,100), range(0.5,1,100)]
    Plots.plot!(xx, [annulus_hat.(0.5, m, x) for x in xx],
        label=[L"H^{K_1, \bullet}_{%$m,1}(x,y)" ""],
        xlabel = L"r",
        ylabel = L"y",
        xlabelfontsize=20, ylabelfontsize=20,
        xtickfontsize=12, ytickfontsize=12,
        linewidth = 2,
        legend= m==0 ? (:bottom) : (:bottomright),
        legendfontsize= 10,
        color=[:orange :orange],
        linestyle=:dash,
        gridlinewidth = 2,
        legend_hfactor = 1.1,
        extra_kwargs=:subplot
    )
    Plots.vline!([-0.5,0.5], label="", color=:black, linewidth=2)
    Plots.savefig("bubble-hats-slice-m-$m.pdf")



    xx = range(-0.5,0.5,200)
    p = Plots.plot(xx, [disk_bubbles.(N, m, xx ./ 0.5) for N in n:2:(n+4)],
        xlabel = L"r",
        ylabel = L"y",
        xlabelfontsize=15, ylabelfontsize=15,
        xtickfontsize=10, ytickfontsize=10,
        linewidth = 2,
        gridlinewidth = 2,
        # legend= m==0 ? (:bottomright) : (:outertopright),
        legend= :outertopright,
        legendfontsize=10,
        # ylim = m==0 ? [-1.3,1.3] : [-1.3,1.3],
        xlim=[-0.54,0.54],
        xticks=[-0.5,-0.25,0,0.25,0.5],
        size=(540,400), #(420,400)
        margin=2Plots.mm,
        label=[L"B^{K_0}_{%$n,%$m,1}(x,y)" L"B^{K_0}_{%$(n+2),%$m,1}(x,y)" L"B^{K_0}_{%$(n+4),%$m,1}(x,y)" L"B^{K_0}_{%$(n+6),%$m,1}(x,y)"]
        )
    Plots.vline!([-0.5,0.5], color=:black, linewidth=2, label="")
    Plots.savefig("bubble-slice-m-$m.pdf")
end
module SparseDiskFEM

using ClassicalOrthogonalPolynomials, PiecewiseOrthogonalPolynomials, RadialPiecewisePolynomials, 
    MultivariateOrthogonalPolynomials, LaTeXStrings, Plots, PyPlot, DelimitedFiles

import RadialPiecewisePolynomials: _getMs_ms_js, Fill, blockedrange
import MultivariateOrthogonalPolynomials: ModalTrav

export plot, cylinder_plot_save, slice_plot, @L_str, writedlm, zplot,
        write_adi_vals,
        list_2_modaltrav, modaltrav_2_list, adi_2_modaltrav, adi_2_list

include("adi.jl")

function _plot(K::Int, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[], vminmax=[]) where T
    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    if vminmax == []
        vmin,vmax = minimum(minimum.(vals)), maximum(maximum.(vals))
    else
        vmin,vmax = vminmax[1], vminmax[2]
    end
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if ρ > 0.0
        ax.set_ylim(ρ,1)
        ax.set_rorigin(0)
        tick_inner_radial = isodd(10*ρ) ? ρ+0.1 : ρ
        ax.set_rticks(tick_inner_radial:0.2:1)
        y_tick_labels = tick_inner_radial:0.2:1
        ax.set_yticklabels(y_tick_labels)
    end

    pc = []
    for k=1:K
        pc = pcolormesh(θs[k], rs[k], vals[k], cmap="bwr", shading="gouraud", norm=norm)
    end

    if ttl != []
        cbar = plt.colorbar(pc, pad=0.1)#, cax = cbar_ax)
        cbar.set_label(ttl)
    end
    display(gcf())
end

function plot(F::FiniteContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[], vminmax=[],K=0) where T
    K = K ==0 ? lastindex(F.points)-1 : K
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl, vminmax=vminmax)
end

function plot(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[], K=0) where T
    K = K ==0 ? lastindex(Z.points)-1 : K
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(Z::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[], K=0) where T
    K = K ==0 ? lastindex(Z.points)-1 : K
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(C::ContinuousZernikeAnnulusElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(C::ContinuousZernikeElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(θs::AbstractVector, rs::AbstractVector, vals::AbstractVector; ρ::T=0.0, ttl=[]) where T
    _plot(1, [[θs[1]; 2π]], rs, [hcat(vals[1], vals[1][:,1])], ρ=ρ, ttl=ttl)
    # _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
end

## slice plot
function slice_plot(iθ::Int, θs, rs, vals, points; ylabel=L"$u(x,y)$", cell_edges=1, title=[])
    rrs = reverse.(rs)
    rvals = [reverse(vals[j][:,iθ]) for j in 1:lastindex(vals)]
    θ = round(θs[1][iθ], digits=4)
    title = title == [] ? L"\theta = %$θ" : title
    p = Plots.plot(rrs,
        rvals,
        # label=["Disk cell" "Annulus cell"],
        linewidth=2,
        ylabel=ylabel,
        xlabel=L"$r$",
        title=title,
        gridlinewidth = 2,
        tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
        legendfontsize=10, titlefontsize=20,
        legend=:none)

    if cell_edges == 1
        vline!(points, color=:black, linewidth=0.8, label="", linestyle=:dash)
    end
    Plots.display(p)
end

function zplot(iθ::Int, rs, z, vals; ttl=[], vminmax=[], xlabel=[], ylabel=[], title=[])
    rrs = reverse.(rs)
    K = lastindex(rrs)

    rvals = [zeros(lastindex(rrs[1]), lastindex(z)) for k in 1:K]
    for i in 1:lastindex(z)
        for k in 1:K
            rvals[k][:,i] = reverse(vals[i][k][:,iθ])
        end
    end


    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if vminmax == []
        vmin,vmax = minimum(minimum.(rvals)), maximum(maximum.(rvals))
    else
        vmin,vmax = vminmax[1], vminmax[2]
    end
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    pc = []
    for k=1:K
        pc = pcolormesh(rrs[k], z, rvals[k]', cmap="bwr", shading="gouraud", norm=norm)
    end

    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if ttl != []
        cbar = plt.colorbar(pc, pad=0.1)#, cax = cbar_ax)
        cbar.set_label(ttl)
    end
    display(gcf())

end

end # module SparseDiskFEM

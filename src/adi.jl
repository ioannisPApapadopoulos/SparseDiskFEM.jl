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
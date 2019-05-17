using Printf

# Source: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        s = t / 1e9
        if s < 60
            value, units = s, "s"
        else
            value, units = (s / 60), "min"
        end
    end
    return string(@sprintf("%.3f", value), " ", units)
end

function Base.zeros(T, ::GPU, g)
    a = CuArray{T}(undef, g.Nx, g.Ny, g.Nz)
    a .= 0
    return a
end

Base.zeros(T, ::CPU, g) = zeros(T, size(g))

# Default to type of Grid
Base.zeros(arch, g::Grid{T}) where T = zeros(T, arch, g)

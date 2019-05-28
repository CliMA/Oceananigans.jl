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

function Base.zeros(T, ::CPU, grid)
    # Starting and ending indices for the offset array.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz

    underlying_data = zeros(T, grid.Tx, grid.Ty, grid.Tz)
    OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

function Base.zeros(T, ::GPU, grid)
    # Starting and ending indices for the offset CuArray.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz

    underlying_data = CuArray{T}(undef, grid.Tx, grid.Ty, grid.Tz)
    OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

# Default to type of Grid
Base.zeros(arch, g::Grid{T}) where T = zeros(T, arch, g)

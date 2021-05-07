module OutputReaders

export FieldTimeSeries

using Base: @propagate_inbounds

using OffsetArrays
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Fields: show_location

import Oceananigans: short_show

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end

struct FieldTimeSeries{X, Y, Z, K, A, T, N, D, G, B} <: AbstractDataField{X, Y, Z, A, G, T, N}
                   data :: D
           architecture :: A
                   grid :: G
    boundary_conditions :: B

    function FieldTimeSeries{X, Y, Z}(backend::K, data::D, arch::A, grid::G, bcs::B) where {X, Y, Z, K, D, A, G, B}
        T = eltype(grid)
        N = ndims(data)
        return new{X, Y, Z, K, A, T, N, D, G, B}(data, arch, grid, bcs)
    end
end

@inline Base.size(fts::FieldTimeSeries) = (size(location(fts), fts.grid)..., size(fts.data, 4))

@propagate_inbounds Base.getindex(f::FieldTimeSeries, i, j, k, n) = f.data[i, j, k, n]

function FieldTimeSeries(filepath, name; architecture=CPU(), backend=InMemory())
    file = jldopen(filepath)

    grid = file["serialized/grid"]
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    LX, LY, LZ = location = file["timeseries/$name/metadata/location"]

    Nt = length(times)
    data_size = size(file["timeseries/$name/0"])
    underlying_data = zeros(data_size..., Nt)
    data = OffsetArray(underlying_data, -Hx, -Hy, -Hz, 0)

    for (n, iter) in enumerate(iterations)
        data.parent[:, :, :, n] .= file["timeseries/$name/$iter"]
    end

    bcs = file["timeseries/$name/metadata/boundary_conditions"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ}(backend, data, architecture, grid, bcs)
end

backend_str(::InMemory) = "InMemory"
backend_str(::OnDisk) = "OnDisk"

short_show(fts::FieldTimeSeries{X, Y, Z, K}) where {X, Y, Z, K} =
    string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K()))} located at $(show_location(fts))")

end # module

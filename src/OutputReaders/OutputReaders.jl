module OutputReaders

export FieldTimeSeries

using OffsetArrays
using JLD2

using Oceananigans.Architectures

using DimensionalData: AbstractDimArray, XDim, YDim, ZDim, X, Y, Z, Ti, formatdims
using Oceananigans.Grids: topology, halo_size, all_x_nodes, all_y_nodes, all_z_nodes, interior_indices

import DimensionalData

struct FieldTimeSeries{X, Y, Z, A, D, Δ, R, G, FT, N, B, M} <: AbstractDimArray{FT, N, Δ, D}
                   data :: D
                   dims :: Δ
                refdims :: R
           architecture :: A
                   grid :: G
    boundary_conditions :: B
               metadata :: M

    function FieldTimeSeries{X, Y, Z}(data::D, dims::Δ, refdims::R, arch::A, grid::G, bcs::B, metadata::M) where {X, Y, Z, D, Δ, R, A, G, B, M}
        FT = eltype(grid)
        N = ndims(data)
        return new{X, Y, Z, A, D, Δ, R, G, FT, N, B, M}(data, dims, refdims, arch, grid, bcs, metadata)
    end
end

infer_indices(dim, default_is, loc, topo, N) = default_is

function infer_indices(dim::Union{XDim, YDim, ZDim}, default_is, loc, topo, N)
    @show typeof(dim)
    is = interior_indices(loc, topo, N)
    is = length(dim) > length(is) ? is : default_is
    return is
end

function DimensionalData.data(f::FieldTimeSeries{LX, LY, LZ, A, <:OffsetArray}) where {LX, LY, LZ, A}
    TX, TY, TZ = topology(f.grid)
    Nx, Ny, Nz = size(f.grid)

    inds = []

    for (d, dim) in enumerate(f.dims)
        if dim isa XDim
            is = interior_indices(LX, TX, Nx)
            is = length(dim) > length(is) ? is : axes(f.data, d)
            push!(inds, is)
        elseif dim isa YDim
            js = interior_indices(LY, TY, Ny)
            js = length(dim) > length(js) ? js : axes(f.data, d)
            push!(inds, js)
        elseif dim isa ZDim
            ks = interior_indices(LZ, TZ, Nz)
            ks = length(dim) > length(ks) ? ks : axes(f.data, d)
            push!(inds, ks)
        else
            push!(inds, axes(f.data, d))
        end
    end

    return view(f.data, inds...)
end

DimensionalData.data(f::FieldTimeSeries{LX, LY, LZ, A, <:SubArray}) where {LX, LY, LZ, A} = f.data
DimensionalData.name(f::FieldTimeSeries) = f.metadata[:name]

@inline DimensionalData.rebuild(f::FieldTimeSeries{X, Y, Z}, data, dims, refdims, name, metadata) where {X, Y, Z} =
    FieldTimeSeries{X, Y, Z}(data, dims, refdims, f.architecture, f.grid, f.boundary_conditions, f.metadata)

function FieldTimeSeries(filepath, name; architecture=CPU())
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

    xs = all_x_nodes(LX, grid)
    ys = all_y_nodes(LY, grid)
    zs = all_z_nodes(LZ, grid)

    x_dim = X(xs)
    y_dim = Y(ys)
    z_dim = Z(zs)
    t_dim = Ti(times)
    dims = (x_dim, y_dim, z_dim, t_dim)

    refdims = ()
    bcs = file["timeseries/$name/metadata/boundary_conditions"]
    metadata = Dict(:name => name)

    close(file)

    return FieldTimeSeries{LX, LY, LZ}(data, formatdims(data, dims), refdims, architecture, grid, bcs, metadata)
end

end # module

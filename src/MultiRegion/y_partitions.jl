using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: MCBC, PBC

const EqualYPartition = YPartition{<:Number}

Base.length(p::YPartition)       = length(p.div)
Base.length(p::EqualYPartition)  = p.div

Base.summary(p::EqualYPartition) = "Equal partitioning in Y with ($(p.div) regions)"
Base.summary(p::YPartition)      = "YPartition with [$(["$(p.div[i]) " for i in 1:length(p)]...)]"

function partition_size(p::EqualYPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert mod(Ny, p.div) == 0
    return Tuple((Nx, Ny ÷ p.div, Nz) for i in 1:length(p))
end

function partition_size(p::YPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert sum(p.div) == Ny
    return Tuple((Nx, p.div[i], Nz) for i in 1:length(p))
end

function partition_extent(p::YPartition, grid)
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    y = divide_direction(y, p)
    return Tuple((x, y = y[i], z = z) for i in 1:length(p))
end

function partition_topology(p::YPartition, grid)
    TX, TY, TZ = topology(grid)
    
    return Tuple((TX, (TY == Periodic ? 
                       FullyConnected : 
                       i == 1 ?
                       RightConnected :
                       i == length(p) ?
                       LeftConnected :
                       FullyConnected), TZ) for i in 1:length(p))
end

divide_direction(x::Tuple, p::EqualYPartition) = 
    Tuple((x[1] + (i-1)*(x[2] - x[1])/length(p), x[1] + i*(x[2] - x[1])/length(p)) for i in 1:length(p))

function divide_direction(x::AbstractArray, p::EqualYPartition) 
    nelem = (length(x)-1)÷length(p)
    return Tuple(x[1+(i-1)*nelem:1+i*nelem] for i in 1:length(p))
end

partition_global_array(a::Field, p::EqualYPartition, args...) = partition_global_array(a.data, p, args...)

function partition_global_array(a::AbstractArray, ::EqualYPartition, local_size, region, arch) 
    idxs = default_indices(length(size(a)))
    offsets = (a.offsets[1], Tuple(0 for i in 1:length(idxs)-1)...)
    return on_architecture(arch, OffsetArray(a[local_size[1]*(region-1)+1+offsets[1]:local_size[1]*region-offsets[1], idxs[2:end]...], offsets...))
end

function partition_global_array(a::OffsetArray, ::EqualYPartition, local_size, region, arch) 
    idxs    = default_indices(length(size(a)))
    offsets = (0, a.offsets[2], Tuple(0 for i in 1:length(idxs)-2)...)
    return on_architecture(arch, OffsetArray(a[idxs[1], local_size[2]*(region-1)+1+offsets[2]:local_size[2]*region-offsets[2], idxs[3:end]...], offsets...))
end

####
#### Global reconstruction utils
####

function reconstruct_size(mrg, p::YPartition)
    Nx = mrg.region_grids[1].Nx
    Ny = sum([grid.Ny for grid in mrg.region_grids.regional_objects]) 
    Nz = mrg.region_grids[1].Nz
    return (Nx, Ny, Nz)
end

function reconstruct_extent(mrg, p::YPartition)
    switch_device!(mrg.devices[1])
    x = cpu_face_constructor_x(mrg.region_grids.regional_objects[1])
    z = cpu_face_constructor_z(mrg.region_grids.regional_objects[1])

    if cpu_face_constructor_y(mrg.region_grids.regional_objects[1]) isa Tuple
        y = (cpu_face_constructor_y(mrg.region_grids.regional_objects[1])[1],
             cpu_face_constructor_y(mrg.region_grids.regional_objects[length(p)])[end])
    else
        y = [cpu_face_constructor_y(mrg.region_grids.regional_objects[1])...]
        for (idx, grid) in enumerate(mrg.region_grids.regional_objects[2:end])
            switch_device!(mrg.devices[idx])
            y = [y..., cpu_face_constructor_y(grid)[2:end]...]
        end
    end
    return (; x, y, z)
end

function reconstruct_global_array(ma::ArrayMRO{T, N}, p::EqualYPartition, arch) where {T, N}
    local_size = size(first(ma.regional_objects))
    global_Ny  = local_size[2] * length(p)
    idxs = default_indices(length(local_size))
    arr_out = zeros(eltype(first(ma.regional_objects)), local_size[1], global_Ny, local_size[3:end]...)

    n = local_size[2]

    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        arr_out[idxs[1], init:fin, idxs[3:end]...] .= on_architecture(CPU(), ma[r])[idxs[1], 1:fin-init+1, idxs[3:end]...]
    end

    return on_architecture(arch, arr_out)
end

function compact_data!(global_field, global_grid, data::MultiRegionObject, p::EqualYPartition)
    Ny = size(global_grid)[2]
    n = Ny / length(p)

    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        interior(global_field)[:, init:fin, :] .= data[r][:, 1:fin-init+1, :]
    end

    fill_halo_regions!(global_field)

    return nothing
end

#####
##### Boundary-specific Utils
#####

const YPartitionConnectivity = Union{RegionalConnectivity{North, South}, RegionalConnectivity{South, North}}

####
#### Global index flattening
####

@inline function displaced_xy_index(i, j, grid, region, p::YPartition)
    j′ = j + grid.Ny * (region - 1) 
    t  = i + (j′ - 1) * grid.Nx
    return t
end

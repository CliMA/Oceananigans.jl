using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: MCBC, PBC

const EqualXPartition = XPartition{<:Number}

Base.length(p::XPartition)       = length(p.div)
Base.length(p::EqualXPartition)  = p.div

Base.summary(p::EqualXPartition) = "Equal partitioning in X with ($(p.div) regions)"
Base.summary(p::XPartition)      = "XPartition with [$(["$(p.div[i]) " for i in 1:length(p)]...)]"

function partition_size(p::EqualXPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert mod(Nx, p.div) == 0 
    return Tuple((Nx ÷ p.div, Ny, Nz) for i in 1:length(p))
end

function partition_size(p::XPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert sum(p.div) == Nx
    return Tuple((p.div[i], Ny, Nz) for i in 1:length(p))
end

function partition_extent(p::XPartition, grid)
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    x = divide_direction(x, p)
    return Tuple((x = x[i], y = y, z = z) for i in 1:length(p))
end

function partition_topology(p::XPartition, grid)
    TX, TY, TZ = topology(grid)
    
    return Tuple(((TX == Periodic ? FullyConnected : i == 1 ?
                                    RightConnected : i == length(p) ?
                                    LeftConnected :
                                    FullyConnected), TY, TZ) for i in 1:length(p))
end

divide_direction(x::Tuple, p::EqualXPartition) =
    Tuple((x[1]+(i-1)*(x[2] - x[1])/length(p), x[1]+i*(x[2] - x[1])/length(p)) for i in 1:length(p))

function divide_direction(x::AbstractArray, p::EqualXPartition)
    nelem = (length(x)-1)÷length(p)
    return Tuple(x[1+(i-1)*nelem:1+i*nelem] for i in 1:length(p))
end

divide_direction(x::Tuple, p::XPartition) =
    Tuple((x[1]+sum(p.div[1:i-1])*(x[2] - x[1])/sum(p.div),
           x[1]+sum(p.div[1:i])  *(x[2] - x[1])/sum(p.div)) for i in 1:length(p))

divide_direction(x::AbstractArray, p::XPartition) =
    Tuple(x[1+sum(p.div[1:i-1]):1+sum(p.div[1:i])] for i in 1:length(p))

partition_global_array(a::Function, args...) = a
partition_global_array(a::Field, p::EqualXPartition, args...) = partition_global_array(a.data, p, args...)

function partition_global_array(a::AbstractArray, ::EqualXPartition, local_size, region, arch)
    idxs = default_indices(length(size(a)))
    return arch_array(arch, a[local_size[1]*(region-1)+1:local_size[1]*region, idxs[2:end]...])
end

function partition_global_array(a::OffsetArray, ::EqualXPartition, local_size, region, arch) 
    idxs    = default_indices(length(size(a)))
    offsets = (a.offsets[1], Tuple(0 for i in 1:length(idxs)-1)...)
    return arch_array(arch, OffsetArray(a[local_size[1]*(region-1)+1+offsets[1]:local_size[1]*region-offsets[1], idxs[2:end]...], offsets...))
end

#####
##### Global reconstruction utils
#####

function reconstruct_size(mrg, p::XPartition)
    Nx = sum([grid.Nx for grid in mrg.region_grids.regional_objects])
    Ny = mrg.region_grids[1].Ny
    Nz = mrg.region_grids[1].Nz
    return (Nx, Ny, Nz)
end

function reconstruct_extent(mrg, p::XPartition)
    switch_device!(mrg.devices[1])
    y = cpu_face_constructor_y(mrg.region_grids.regional_objects[1])
    z = cpu_face_constructor_z(mrg.region_grids.regional_objects[1])

    if cpu_face_constructor_x(mrg.region_grids.regional_objects[1]) isa Tuple
        x = (cpu_face_constructor_x(mrg.region_grids.regional_objects[1])[1],
             cpu_face_constructor_x(mrg.region_grids.regional_objects[length(p)])[end])
    else
        x = [cpu_face_constructor_x(mrg.region_grids.regional_objects[1])...]
        for (idx, grid) in enumerate(mrg.region_grids.regional_objects[2:end])
            switch_device!(mrg.devices[idx])
            x = [x..., cpu_face_constructor_x(grid)[2:end]...]
        end
    end

    return (; x, y, z)
end

const FunctionMRO    = MultiRegionObject{<:Tuple{Vararg{T}}} where T<:Function
const ArrayMRO{T, N} = MultiRegionObject{<:Tuple{Vararg{A}}} where A<:AbstractArray{T, N} where {T, N}

reconstruct_global_array(ma::FunctionMRO, args...) = ma.regional_objects[1]

function reconstruct_global_array(ma::ArrayMRO{T, N}, p::EqualXPartition, arch) where {T, N}
    local_size = size(first(ma.regional_objects))
    global_Nx  = local_size[1] * length(p)
    idxs = default_indices(length(local_size))
    arr_out = zeros(eltype(first(ma.regional_objects)), global_Nx, local_size[2:end]...)

    n = local_size[1]

    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        arr_out[init:fin, idxs[2:end]...] .= arch_array(CPU(), ma[r])[1:fin-init+1, idxs[2:end]...]
    end

    return arch_array(arch, arr_out)
end

function compact_data!(global_field, global_grid, data::MultiRegionObject, p::EqualXPartition)
    Nx = size(global_grid)[1]
    n = Nx / length(p)

    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        interior(global_field)[init:fin, :, :] .= data[r][1:fin-init+1, :, :]
    end

    fill_halo_regions!(global_field)

    return nothing
end

#####
##### Boundary-specific Utils
#####

const XPartitionConnectivity = Union{RegionalConnectivity{East, West}, RegionalConnectivity{West, East}}

####
#### Global index flattening
####

@inline function displaced_xy_index(i, j, grid, region, p::XPartition)
    i′ = i + grid.Nx * (region - 1)
    t  = i′ + (j - 1) * grid.Nx * length(p)
    return t
end

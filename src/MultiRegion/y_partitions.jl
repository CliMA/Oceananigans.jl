using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: CBC, PBC
using Oceananigans.Fields: extract_field_data

struct YPartition{N} <: AbstractPartition
    div :: N
    function YPartition(sizes) 
        if length(sizes) > 1 && all(y -> y == sizes[1], sizes)
            sizes = length(sizes)
        end
        return new{typeof(sizes)}(sizes)
    end
end

const EqualYPartition = YPartition{<:Number}

length(p::EqualYPartition) = p.div
length(p::YPartition)      = length(p.div)

Base.summary(p::EqualYPartition) = "Equal partitioning in Y ($(p.div) regions)"
Base.summary(p::YPartition)      = "partitioning in Y [$(["$(p.div[i]) " for i in 1:length(p)]...)]"

function partition_size(p::EqualYPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert mod(Ny, p.div) == 0 
    return Tuple((Nx, Ny ÷ p.div, Nz) for i in 1:length(p))
end

function partition_size(p::YPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert sum(p.div) != Ny
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
    Tuple((x[1]+(i-1)*(x[2] - x[1])/length(p), x[1]+i*(x[2] - x[1])/length(p)) for i in 1:length(p))

function divide_direction(x::AbstractArray, p::EqualYPartition) 
    nelem = (length(x)-1)÷length(p)
    return Tuple(x[1+(i-1)*nelem:1+i*nelem] for i in 1:length(p))
end

function reconstruct_size(mrg, p::YPartition)
    Nx = mrg.region_grids[1].Nx
    Ny = sum([grid.Ny for grid in mrg.region_grids.regions]) 
    Nz = mrg.region_grids[1].Nz
    return (Nx, Ny, Nz)
end

function reconstruct_extent(mrg, p::YPartition)
    switch_device!(mrg.devices[1])
    x = cpu_face_constructor_x(mrg.region_grids.regions[1])
    z = cpu_face_constructor_z(mrg.region_grids.regions[1])

    if cpu_face_constructor_y(mrg.region_grids.regions[1]) isa Tuple
        y = (cpu_face_constructor_y(mrg.region_grids.regions[1])[1], 
             cpu_face_constructor_y(mrg.region_grids.regions[length(p)])[end])
    else
        y = [cpu_face_constructor_y(mrg.region_grids.regions[1])...]
        for (idx, grid) in enumerate(mrg.region_grids.regions)
            switch_device!(mrg.devices[idx])
            x = [x..., cpu_face_constructor_y(grid)[2:end]...]
        end
    end
    return (; x = x, y = y, z = z)
end

inject_west_boundary(region, p::YPartition, bc) = bc 
inject_east_boundary(region, p::YPartition, bc) = bc

function inject_south_boundary(region, p::YPartition, global_bc) 
    if region == 1
        typeof(global_bc) <: Union{CBC, PBC} ?  
                bc = CommunicationBoundaryCondition((rank = region, from_rank = length(p))) : 
                bc = global_bc
    else
        bc = CommunicationBoundaryCondition((rank = region, from_rank = region - 1))
    end
    return bc
end

function inject_north_boundary(region, p::YPartition, global_bc) 
    if region == length(p)
        typeof(global_bc) <: Union{CBC, PBC} ?  
                bc = CommunicationBoundaryCondition((rank = region, from_rank = 1)) : 
                bc = global_bc
    else
        bc = CommunicationBoundaryCondition((rank = region, from_rank = region + 1))
    end
    return bc
end

function partition_global_array(a::AbstractArray, ::EqualYPartition, grid, local_size, region, arch) 
    idxs = default_indices(length(size(a)))
    return arch_array(arch, a[idxs[1], local_size[2]*(region-1)+1:local_size[2]*region, idxs[3:end]...])
end

function reconstruct_global_array(ma::ArrayMRO{T, N}, p::EqualYPartition, arch) where {T, N}
    local_size = size(first(ma.regions))
    global_Nx  = local_size[2] * length(p)
    idxs = default_indices(length(local_size))
    arr_out = zeros(eltype(first(ma.regions)), global_Nx, local_size[2:end]...)
    n = local_size[2]
    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        arr_out[idxs[1], init:fin, idxs[3:end]...] .= arch_array(CPU(), ma[r])[idxs[1], 1:fin-init+1, idxs[3:end]...]
    end

    return arch_array(arch, arr_out)
end

function compact_data!(global_field, global_grid, data::MultiRegionObject, p::EqualYPartition)
    Nx, Ny, Nz = size(global_grid)
    n = Ny / length(p)
    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        interior(global_field)[:, init:fin, :] .= data[r][:, 1:fin-init+1, :]
    end
end

@inline function displaced_xy_index(i, j, grid, region, p::YPartition)
    j′ = j + grid.Ny * (region - 1) 
    t  = i + (j′ - 1) * grid.Nx
    return t
end

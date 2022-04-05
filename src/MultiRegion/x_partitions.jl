using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: CBC, PBC
using Oceananigans.Fields: extract_field_data

struct XPartition{N} <: AbstractPartition
    div :: N
    function XPartition(sizes) 
        if length(sizes) > 1 && all(y -> y == sizes[1], sizes)
            sizes = length(sizes)
        end
        return new{typeof(sizes)}(sizes)
    end
end

const EqualXPartition = XPartition{<:Number}

length(p::EqualXPartition) = p.div
length(p::XPartition)      = length(p.div)

Base.summary(p::EqualXPartition) = "Equal partitioning in X ($(p.div) regions)"
Base.summary(p::XPartition)      = "partitioning in X [$(["$(p.div[i]) " for i in 1:length(p)]...)]"

function partition_size(p::EqualXPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert mod(Nx, p.div) == 0 
    return Tuple((Nx ÷ p.div, Ny, Nz) for i in 1:length(p))
end

function partition_size(p::XPartition, grid)
    Nx, Ny, Nz = size(grid)
    @assert sum(p.div) != Nx
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
    
    return Tuple(((TX == Periodic ? 
                   FullyConnected : 
                   i == 1 ?
                   RightConnected :
                   i == length(p) ?
                   LeftConnected :
                   FullyConnected), TY, TZ) for i in 1:length(p))
end

divide_direction(x::Tuple, p::EqualXPartition) = 
    Tuple((x[1]+(i-1)*(x[2] - x[1])/length(p), x[1]+i*(x[2] - x[1])/length(p)) for i in 1:length(p))

function divide_direction(x::AbstractArray, p::EqualXPartition) 
    nelem = (length(x)-1)÷length(p)
    return Tuple(x[1+(i-1)*nelem:1+i*nelem] for i in 1:length(p))
end

function reconstruct_size(mrg, p::XPartition)
    Ny = mrg.region_grids[1].Ny
    Nz = mrg.region_grids[1].Nz
    Nx = sum([grid.Nx for grid in mrg.region_grids.regions])
    return (Nx, Ny, Nz)
end

function reconstruct_extent(mrg, p::XPartition)
    y = cpu_face_constructor_y(mrg.region_grids.regions[1])
    z = cpu_face_constructor_z(mrg.region_grids.regions[1])

    if cpu_face_constructor_x(mrg.region_grids.regions[1]) isa Tuple
        x = (cpu_face_constructor_x(mrg.region_grids.regions[1])[1], 
             cpu_face_constructor_x(mrg.region_grids.regions[length(p)])[end])
    else
        x = [cpu_face_constructor_x(mrg.region_grids.regions[1])...]
        for grid in mrg.region_grids.regions
            x = [x..., cpu_face_constructor_x(grid)[2:end]...]
        end
    end
    return (; x = x, y = y, z = z)
end

inject_south_boundary(region, p::XPartition, bc) = bc
inject_north_boundary(region, p::XPartition, bc) = bc

function inject_west_boundary(region, p::XPartition, global_bc) 
    if region == 1
        typeof(global_bc) <: Union{CBC, PBC} ?  
                bc = CommunicationBoundaryCondition((rank = region, from_rank = length(p))) : 
                bc = global_bc
    else
        bc = CommunicationBoundaryCondition((rank = region, from_rank = region - 1))
    end
    return bc
end

function inject_east_boundary(region, p::XPartition, global_bc) 
    if region == length(p)
        typeof(global_bc) <: Union{CBC, PBC} ?  
                bc = CommunicationBoundaryCondition((rank = region, from_rank = 1)) : 
                bc = global_bc
    else
        bc = CommunicationBoundaryCondition((rank = region, from_rank = region + 1))
    end
    return bc
end

partition_immersed_boundary(b, args...) = getname(b)(partition_global_array(getproperty(b, propertynames(b)[1]), args...))

partition_global_array(a::Function, args...) = a

function partition_global_array(a::AbstractArray, ::EqualXPartition, grid, local_size, region, arch) 
    idxs = default_indices(length(local_size)-1)
    return arch_array(arch, a[local_size[1]*(region-1)+1:local_size[1]*region, idxs...])
end

function compact_data!(global_field, global_grid, data::MultiRegionObject, p::EqualXPartition)
    Nx, Ny, Nz = size(global_grid)
    n = Nx / length(p)
    for r = 1:length(p)
        init = Int(n * (r - 1) + 1)
        fin  = Int(n * r)
        global_field.data[init:fin, 1:Ny, 1:Nz] .= data[r]
    end
end

@inline function displaced_xy_index(i, j, grid, region, p::EqualXPartition)
    i′ = i + grid.Nx * (region - 1) 
    return i′ + (j - 1) * grid.Nx * length(p)
end

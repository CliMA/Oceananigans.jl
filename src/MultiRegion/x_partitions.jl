using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z

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

inject_south_boundary(region, p::XPartition, bc) = bc
inject_north_boundary(region, p::XPartition, bc) = bc

function inject_west_boundary(region, p::XPartition, global_bc) 
    if region == 1
        typeof(global_bc) <: BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing} ?  
                bc = HaloBoundaryCondition((rank = region, from_rank = length(p))) : 
                bc = global_bc
    else
        bc = HaloBoundaryCondition((rank = region, from_rank = region - 1))
    end
    return bc
end

function inject_east_boundary(region, p::XPartition, global_bc) 
    if region == length(p)
        typeof(global_bc) <: BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing} ?  
                bc = HaloBoundaryCondition((rank = region, from_rank = 1)) : 
                bc = global_bc
    else
        bc = HaloBoundaryCondition((rank = region, from_rank = region + 1))
    end
    return bc
end
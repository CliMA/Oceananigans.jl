using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.Grids: R_Earth, validate_lat_lon_grid_args, generate_coordinate, with_precomputed_metrics

import Oceananigans.Grids: zeros
import Oceananigans.Architectures: child_architecture

import Oceananigans.DistributedComputations: 
    partition_coordinate, 
    assemble_coordinate, 
    inject_halo_communication_boundary_conditions,
    concatenate_local_sizes,
    barrier!,
    all_reduce

child_architecture(grid::ShardedGrid) = child_architecture(architecture(grid))

# Coordinates do not need partitioning on a `Distributed{<:ReactantState}` (sharded) architecture
partition_coordinate(c::Tuple,          n, ::ShardedDistributed, dim) = c
partition_coordinate(c::AbstractVector, n, ::ShardedDistributed, dim) = c

# Same thing for assembling the coordinate, it is already represented as a global array
assemble_coordinate(c::Tuple,          n, ::ShardedDistributed, dim) = c
assemble_coordinate(c::AbstractVector, n, ::ShardedDistributed, dim) = c

# Boundary conditions should not need to change
inject_halo_communication_boundary_conditions(field_bcs, rank, ::Reactant.Sharding.Mesh, topology) = field_bcs

# Local sizes are equal to global sizes for a sharded architecture
concatenate_local_sizes(local_size, ::ShardedDistributed) = local_size

# We assume everything is already synchronized for a sharded architecture
barrier!(::ShardedDistributed) = nothing

# Reductions are handled by the Sharding framework
all_reduce(op, val, ::ShardedDistributed) = val

# No need for partitioning and assembling of arrays supposedly
partition(A::AbstractArray, ::ShardedDistributed, local_size) = A
construct_global_array(A::AbstractArray, ::ShardedDistributed, local_size) = A

function Oceananigans.LatitudeLongitudeGrid(arch::ShardedDistributed,
                                            FT::DataType = Oceananigans.defaults.FloatType;
                                            size,
                                            longitude = nothing,
                                            latitude = nothing,
                                            z = nothing,
                                            radius = Oceananigans.Grids.R_Earth,
                                            topology = nothing,
                                            precompute_metrics = true,
                                            halo = nothing)

    topology, size, halo, latitude, longitude, z, precompute_metrics =
        validate_lat_lon_grid_args(topology, size, halo, FT, latitude, longitude, z, precompute_metrics)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo
    TX, TY, TZ = topology

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, longitude, :longitude, 1, arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topology, size, halo, latitude,  :latitude,  2, arch)
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z,         :z,         3, arch)

    # We build the grid on the CPU and then we move it to ReactantState
    grid = LatitudeLongitudeGrid{TX, TY, TZ}(CPU(),
                                             Nλ, Nφ, Nz,
                                             Hλ, Hφ, Hz,
                                             Lλ, Lφ, Lz,
                                             Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                             Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                             z, # Intentionally not sharded
                                             (nothing for i=1:10)..., FT(radius))

    # Sharding of metrics and coordinates does not seem to 
    # work. However, a sharded grid might work also with non-sharded metrics?
    # TODO: Remove this to shard metrics and coordinated
    φsharding = Sharding.NoSharding()
    λsharding = Sharding.NoSharding()
    λmetric_sharding  = Sharding.NoSharding()
    φmetric_sharding  = Sharding.NoSharding()
    xzmetric_sharding = Sharding.NoSharding()
    ymetric_sharding  = Sharding.NoSharding()

    #= Uncomment this to allow sharding of the metrics
    # Extracting the local range
    xsharding  = Sharding.DimsSharding(arch.connectivity, (1,  ), (:x,   )) # X Stencil sharding
    ysharding  = Sharding.DimsSharding(arch.connectivity, (2,  ), (:y,   )) # Y Stencil sharding
    xysharding = Sharding.DimsSharding(arch.connectivity, (1, 2), (:x, :y)) # XY Pencil sharding

    λmetric_sharding = ndims(Δλᶜᵃᵃ) == 1 ? xsharding : Sharding.NoSharding()
    φmetric_sharding = ndims(Δφᵃᶜᵃ) == 1 ? ysharding : Sharding.NoSharding()

    λsharding = parent(λᶜᵃᵃ) isa StepRangeLen ? Sharding.NoSharding() : xsharding
    φsharding = parent(φᵃᶜᵃ) isa StepRangeLen ? Sharding.NoSharding() : ysharding
    
    # y metrics are either 1D or a number, while x and z metrics are either 2D or 1D
    ymetric_sharding  = ndims(Δφᵃᶜᵃ) == 1 ? ysharding  : Sharding.NoSharding() # Will this work?
    xzmetric_sharding = ndims(Δλᶜᵃᵃ) == 1 ? xysharding : xysharding
    =#

    # Sharding common metricd
    Δλᶠᵃᵃ = Reactant.to_rarray(grid.Δλᶠᵃᵃ; sharding=λmetric_sharding)
    Δλᶜᵃᵃ = Reactant.to_rarray(grid.Δλᶜᵃᵃ; sharding=λmetric_sharding)
    λᶠᵃᵃ  = Reactant.to_rarray(grid.λᶠᵃᵃ ; sharding=λsharding)
    λᶜᵃᵃ  = Reactant.to_rarray(grid.λᶜᵃᵃ ; sharding=λsharding)
    Δφᵃᶠᵃ = Reactant.to_rarray(grid.Δφᵃᶠᵃ; sharding=φmetric_sharding)
    Δφᵃᶜᵃ = Reactant.to_rarray(grid.Δφᵃᶜᵃ; sharding=φmetric_sharding)
    φᵃᶠᵃ  = Reactant.to_rarray(grid.φᵃᶠᵃ ; sharding=φsharding)
    φᵃᶜᵃ  = Reactant.to_rarray(grid.φᵃᶜᵃ ; sharding=φsharding)
    z     = Reactant.to_rarray(grid.z) # Intentionally not sharded

    if !precompute_metrics
        return LatitudeLongitudeGrid{TX, TY, TZ}(arch,
                                                 grid.Nx, grid.Ny, grid.Nz,
                                                 grid.Hx, grid.Hy, grid.Hz,
                                                 grid.Lx, grid.Ly, grid.Lz,
                                                 Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                 Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                 z, # Intentionally not sharded
                                                 (nothing for i=1:10)..., 
                                                 grid.radius)
    else
        grid = with_precomputed_metrics(grid) 

        return LatitudeLongitudeGrid{TX, TY, TZ}(arch,
                                                grid.Nx, grid.Ny, grid.Nz,
                                                grid.Hx, grid.Hy, grid.Hz,
                                                grid.Lx, grid.Ly, grid.Lz,
                                                Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                z, # Intentionally not sharded
                                                Reactant.to_rarray(grid.Δxᶜᶜᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Δxᶠᶜᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Δxᶜᶠᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Δxᶠᶠᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Δyᶠᶜᵃ; sharding=ymetric_sharding),
                                                Reactant.to_rarray(grid.Δyᶜᶠᵃ; sharding=ymetric_sharding),
                                                Reactant.to_rarray(grid.Azᶜᶜᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Azᶠᶜᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Azᶜᶠᵃ; sharding=xzmetric_sharding),
                                                Reactant.to_rarray(grid.Azᶠᶠᵃ; sharding=xzmetric_sharding),
                                                grid.radius)
    end
end

# This mostly exists for future where we will assemble data from multiple workers
# to construct the grid
function TripolarGrid(arch::ShardedDistributed,
                      FT::DataType=Float64;
                      halo=(4, 4, 4), kwargs...)

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    global_size = size(global_grid)

    # Extracting the local range
    sharding = Sharding.DimsSharding(arch.connectivity, (1, 2), (:x, :y))

    # Needed for partitial array assembly
    # device_to_array_slices = Reactant.sharding_to_array_slices(sharding, global_size)
    irange = Colon()
    jrange = Colon()
    FT = eltype(global_grid)

    # Partitioning the Coordinates
    λᶠᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶠᶠᵃ, irange, jrange)
    φᶠᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶠᶠᵃ, irange, jrange)
    λᶠᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶠᶜᵃ, irange, jrange)
    φᶠᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶠᶜᵃ, irange, jrange)
    λᶜᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶜᶠᵃ, irange, jrange)
    φᶜᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶜᶠᵃ, irange, jrange)
    λᶜᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶜᶜᵃ, irange, jrange)
    φᶜᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶜᶜᵃ, irange, jrange)

    # # Partitioning the Metrics
    Δxᶜᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶜᶜᵃ, irange, jrange)
    Δxᶠᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶠᶜᵃ, irange, jrange)
    Δxᶜᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶜᶠᵃ, irange, jrange)
    Δxᶠᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶠᶠᵃ, irange, jrange)
    Δyᶜᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶜᶜᵃ, irange, jrange)
    Δyᶠᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶠᶜᵃ, irange, jrange)
    Δyᶜᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶜᶠᵃ, irange, jrange)
    Δyᶠᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶠᶠᵃ, irange, jrange)
    Azᶜᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶜᶜᵃ, irange, jrange)
    Azᶠᶜᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶠᶜᵃ, irange, jrange)
    Azᶜᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶜᶠᵃ, irange, jrange)
    Azᶠᶠᵃ = OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶠᶠᵃ, irange, jrange)

    grid = OrthogonalSphericalShellGrid{Periodic,RightConnected,Bounded}(arch,
        global_size...,
        halo...,
        convert(FT, global_grid.Lz),
        Reactant.to_rarray(λᶜᶜᵃ; sharding),
        Reactant.to_rarray(λᶠᶜᵃ; sharding),
        Reactant.to_rarray(λᶜᶠᵃ; sharding),
        Reactant.to_rarray(λᶠᶠᵃ; sharding),
        Reactant.to_rarray(φᶜᶜᵃ; sharding),
        Reactant.to_rarray(φᶠᶜᵃ; sharding),
        Reactant.to_rarray(φᶜᶠᵃ; sharding),
        Reactant.to_rarray(φᶠᶠᵃ; sharding),
        Reactant.to_rarray(global_grid.z), # Intentionally not sharded
        Reactant.to_rarray(Δxᶜᶜᵃ; sharding),
        Reactant.to_rarray(Δxᶠᶜᵃ; sharding),
        Reactant.to_rarray(Δxᶜᶠᵃ; sharding),
        Reactant.to_rarray(Δxᶠᶠᵃ; sharding),
        Reactant.to_rarray(Δyᶜᶜᵃ; sharding),
        Reactant.to_rarray(Δyᶠᶜᵃ; sharding),
        Reactant.to_rarray(Δyᶜᶠᵃ; sharding),
        Reactant.to_rarray(Δyᶠᶠᵃ; sharding),
        Reactant.to_rarray(Azᶜᶜᵃ; sharding),
        Reactant.to_rarray(Azᶠᶜᵃ; sharding),
        Reactant.to_rarray(Azᶜᶠᵃ; sharding),
        Reactant.to_rarray(Azᶠᶠᵃ; sharding),
        convert(FT, global_grid.radius),
        global_grid.conformal_mapping)

    return grid
end

function Oceananigans.Grids.zeros(arch::ShardedDistributed, FT, global_sz...)
    # TODO: still need a "pre-sharded" zeros function
    cpu_zeros = zeros(CPU(), FT, global_sz...)
    sharding = Sharding.DimsSharding(arch.connectivity, (1, 2, 3), (:x, :y, :z))
    reactant_zeros = Reactant.to_rarray(cpu_zeros; sharding)
    return reactant_zeros 
end


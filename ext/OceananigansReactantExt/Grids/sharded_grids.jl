using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.Grids: R_Earth, validate_lat_lon_grid_args, generate_coordinate, with_precomputed_metrics

import Oceananigans.Grids: zeros, StaticVerticalDiscretization
import Oceananigans.Architectures: child_architecture

import Oceananigans.DistributedComputations: 
    partition_coordinate, 
    assemble_coordinate, 
    inject_halo_communication_boundary_conditions,
    concatenate_local_sizes,
    barrier!,
    all_reduce,
    all_reduce!,
    reconstruct_global_topology

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
all_reduce(op,  val, ::ShardedDistributed) = val
all_reduce!(op, val, ::ShardedDistributed) = val

# No need for partitioning and assembling of arrays supposedly
partition(A::AbstractArray, ::ShardedDistributed, local_size) = A
construct_global_array(A::AbstractArray, ::ShardedDistributed, local_size) = A

reconstruct_global_topology(topo, R, r, r1, r2, ::ShardedDistributed) = topo

# A function to shard the z-direction (needs to be replicated around 
# TODO: add a method for `MutableVerticalDiscretization`
function sharded_z_direction(z::StaticVerticalDiscretization; sharding = Sharding.NoSharding()) 
    
    cᵃᵃᶠ = parent(z.cᵃᵃᶠ) isa StepRangeLen ? z.cᵃᵃᶠ : Reactant.to_rarray(z.cᵃᵃᶠ; sharding)
    cᵃᵃᶜ = parent(z.cᵃᵃᶜ) isa StepRangeLen ? z.cᵃᵃᶜ : Reactant.to_rarray(z.cᵃᵃᶜ; sharding)

    Δᵃᵃᶠ = Reactant.to_rarray(z.Δᵃᵃᶠ; sharding)
    Δᵃᵃᶜ = Reactant.to_rarray(z.Δᵃᵃᶜ; sharding)

    return StaticVerticalDiscretization(cᵃᵃᶠ, cᵃᵃᶜ, Δᵃᵃᶠ, Δᵃᵃᶜ)
end

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

    grid = with_precomputed_metrics(grid) 

    # Extracting the local range
    xsharding  = Sharding.DimsSharding(arch.connectivity, (1,  ), (:x,   )) # X Stencil sharding
    ysharding  = Sharding.DimsSharding(arch.connectivity, (1,  ), (:y,   )) # Y Stencil sharding
    xysharding = Sharding.DimsSharding(arch.connectivity, (1, 2), (:x, :y)) # XY Pencil sharding

    # Copying the z coordinate to all the devices: we pass a NamedSharding of `nothing`s
    # (a NamedSharding of nothings represents a copy to all devices)
    # ``1'' here is the maximum number of dimensions of the fields of ``z''
    replicate = Sharding.NamedSharding(arch.connectivity, ntuple(Returns(nothing), 1)) 

    λsharding = parent(λᶜᵃᵃ) isa StepRangeLen ? Sharding.NoSharding() : xsharding
    φsharding = parent(φᵃᶜᵃ) isa StepRangeLen ? Sharding.NoSharding() : ysharding
    
    # y metrics are either 1D or a number, while x and z metrics are either 2D or 1D
    xzmetric_sharding = ndims(grid.Δxᶜᶜᵃ) == 1 ? ysharding : xysharding 
    
    # Sharding common metricd
    Δλᶠᵃᵃ = Reactant.to_rarray(grid.Δλᶠᵃᵃ; sharding=xsharding)
    Δλᶜᵃᵃ = Reactant.to_rarray(grid.Δλᶜᵃᵃ; sharding=xsharding)
    λᶠᵃᵃ  = Reactant.to_rarray(grid.λᶠᵃᵃ ; sharding=λsharding)
    λᶜᵃᵃ  = Reactant.to_rarray(grid.λᶜᵃᵃ ; sharding=λsharding)
    Δφᵃᶠᵃ = Reactant.to_rarray(grid.Δφᵃᶠᵃ; sharding=ysharding)
    Δφᵃᶜᵃ = Reactant.to_rarray(grid.Δφᵃᶜᵃ; sharding=ysharding)
    φᵃᶠᵃ  = Reactant.to_rarray(grid.φᵃᶠᵃ ; sharding=φsharding)
    φᵃᶜᵃ  = Reactant.to_rarray(grid.φᵃᶜᵃ ; sharding=φsharding)
    z     = sharded_z_direction(grid.z; sharding=replicate) # Intentionally not sharded

    Δxᶜᶜᵃ = Reactant.to_rarray(grid.Δxᶜᶜᵃ; sharding=xzmetric_sharding)
    Δxᶠᶜᵃ = Reactant.to_rarray(grid.Δxᶠᶜᵃ; sharding=xzmetric_sharding)
    Δxᶜᶠᵃ = Reactant.to_rarray(grid.Δxᶜᶠᵃ; sharding=xzmetric_sharding)
    Δxᶠᶠᵃ = Reactant.to_rarray(grid.Δxᶠᶠᵃ; sharding=xzmetric_sharding)
    Δyᶠᶜᵃ = Reactant.to_rarray(grid.Δyᶠᶜᵃ; sharding=ysharding)
    Δyᶜᶠᵃ = Reactant.to_rarray(grid.Δyᶜᶠᵃ; sharding=ysharding)
    Azᶜᶜᵃ = Reactant.to_rarray(grid.Azᶜᶜᵃ; sharding=xzmetric_sharding)
    Azᶠᶜᵃ = Reactant.to_rarray(grid.Azᶠᶜᵃ; sharding=xzmetric_sharding)
    Azᶜᶠᵃ = Reactant.to_rarray(grid.Azᶜᶠᵃ; sharding=xzmetric_sharding)
    Azᶠᶠᵃ = Reactant.to_rarray(grid.Azᶠᶠᵃ; sharding=xzmetric_sharding)

    if !precompute_metrics
        throw(ArgumentError("On-the-fly metric computation is not supported on sharded architectures."))
    end

    return LatitudeLongitudeGrid{TX, TY, TZ}(arch,
                                             grid.Nx, grid.Ny, grid.Nz,
                                             grid.Hx, grid.Hy, grid.Hz,
                                             grid.Lx, grid.Ly, grid.Lz,
                                             Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                             Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                             z, # Intentionally not sharded
                                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                             Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                             Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                             grid.radius)
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

    # Copying the z coordinate to all the devices: we pass a NamedSharding of `nothing`s
    # (a NamedSharding of nothings represents a copy to all devices)
    # ``1'' here is the maximum number of dimensions of the fields of ``z''
    replicate = Sharding.NamedSharding(arch.connectivity, ntuple(Returns(nothing), 1)) 

    grid = OrthogonalSphericalShellGrid{Periodic, RightFolded, Bounded}(arch,
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
        sharded_z_direction(global_grid.z; sharding=replicate), # Replicating on all devices
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


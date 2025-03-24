import Oceananigans.DistributedComputations: partition_coordinate, assemble_coordinate

# Coordinates do not need partitioning on a `Distributed{<:ReactantState}` (sharded) architecture
partition_coordinate(c::Tuple,          n, ::Oceananigans.Distributed{<:ReactantState}, dim) = c
partition_coordinate(c::AbstractVector, n, ::Oceananigans.Distributed{<:ReactantState}, dim) = c

# Same thing for assembling the coordinate, it is already represented as a global array
assemble_coordinate(c::Tuple,          n, ::Oceananigans.Distributed{<:ReactantState}, dim) = c
assemble_coordinate(c::AbstractVector, n, ::Oceananigans.Distributed{<:ReactantState}, dim) = c

# This mostly exists for future where we will assemble data from multiple workers
# to construct the grid
function Oceananigans.TripolarGrid(arch::Oceananigans.Distributed{<:ReactantState},
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

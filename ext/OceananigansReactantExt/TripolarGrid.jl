# Move to Reactant
struct TreeSharding{S} <: Sharding.AbstractSharding
    sharding::S
end

Sharding.is_sharded(sharding::TreeSharding) = true

Sharding.ndevices(sharding::TreeSharding) = Sharding.ndevices(sharding.sharding)
Sharding.shard_type(::Type{TreeSharding{S}}, N) where {S} = Sharding.shard_type(S, N)

Base.getproperty(t::TreeSharding, x) = t
function Base.getproperty(t::TreeSharding, x::Symbol)
    x == :sharding && return getfield(t, :sharding)
    return t
end

function (sharding::TreeSharding)(
    client::Reactant.XLA.AbstractClient, device, x::Union{AbstractArray,Number}
)
    return sharding.sharding(client, device, x)
end

function Oceananigans.TripolarGrid(arch::Oceananigans.Distributed{<:ReactantState},
    FT::DataType=Float64;
    halo=(4, 4, 4), kwargs...)
    # workers = DistributedComputations.ranks(arch.partition)
    px = ifelse(isnothing(arch.partition.x), 1, arch.partition.x)
    py = ifelse(isnothing(arch.partition.y), 1, arch.partition.y)

    # Check that partitioning in x is correct:
    try
        if isodd(px) && (px != 1)
            throw(ArgumentError("Only even partitioning in x is supported with the TripolarGrid"))
        end
    catch
        throw(ArgumentError("The x partition $(px) is not supported. The partition in x must be an even number. "))
    end

    # a slab decomposition in x is not supported
    if px != 1 && py == 1
        throw(ArgumentError("A x-only partitioning is not supported with the TripolarGrid.\n
                             Please, use a y partitioning configuration or a x-y pencil \
                             partitioning."))
    end

    Hx, Hy, Hz = halo

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    Nx, Ny, Nz = global_size = size(global_grid)

    # Splitting the grid manually
    # lsize = DistributedComputations.local_size(arch, global_size)

    # Extracting the local range
    mesh = arch.connectivity
    devices = arch.devices
    sharding = TreeSharding(Sharding.DimsSharding(mesh, (1, 2), (:x, :y)))

    # # XXX: cleanup api
    # Needed for partitial array assembly
    # hlo_sharding = Sharding.generate_hlo_sharding_from_tensor_attribute(sharding)
    # condensed_op_sharding = convert(
    #     Reactant.XLA.CondensedOpSharding, hlo_sharding.hlo_sharding
    # )
    # device_to_array_slices, needs_padding = Reactant.XLA.sharding_to_concrete_array_indices(
    #     condensed_op_sharding, global_size, hlo_sharding.mesh.logical_device_ids
    # )

    # @show device_to_array_slices

    # if needs_padding
    #     error("XXXX: Padding not implemented")
    # end

    irange = Colon()
    jrange = Colon()
    FT = eltype(global_grid)

    # Partitioning the Coordinates
    λᶠᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶠᶠᵃ, irange, jrange)
    λᶠᶠᵃ = Reactant.to_rarray(map(FT, λᶠᶠᵃ); sharding)
    φᶠᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶠᶠᵃ, irange, jrange)
    φᶠᶠᵃ = Reactant.to_rarray(map(FT, φᶠᶠᵃ); sharding)
    λᶠᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶠᶜᵃ, irange, jrange)
    λᶠᶜᵃ = Reactant.to_rarray(map(FT, λᶠᶜᵃ); sharding)
    φᶠᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶠᶜᵃ, irange, jrange)
    φᶠᶜᵃ = Reactant.to_rarray(map(FT, φᶠᶜᵃ); sharding)
    λᶜᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶜᶠᵃ, irange, jrange)
    λᶜᶠᵃ = Reactant.to_rarray(map(FT, λᶜᶠᵃ); sharding)
    φᶜᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶜᶠᵃ, irange, jrange)
    φᶜᶠᵃ = Reactant.to_rarray(map(FT, φᶜᶠᵃ); sharding)
    λᶜᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :λᶜᶜᵃ, irange, jrange)
    λᶜᶜᵃ = Reactant.to_rarray(map(FT, λᶜᶜᵃ); sharding)
    φᶜᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :φᶜᶜᵃ, irange, jrange)
    φᶜᶜᵃ = Reactant.to_rarray(map(FT, φᶜᶜᵃ); sharding)

    # # Partitioning the Metrics
    Δxᶜᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶜᶜᵃ, irange, jrange)
    Δxᶜᶜᵃ = Reactant.to_rarray(map(FT, Δxᶜᶜᵃ); sharding)
    Δxᶠᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶠᶜᵃ, irange, jrange)
    Δxᶠᶜᵃ = Reactant.to_rarray(map(FT, Δxᶠᶜᵃ); sharding)
    Δxᶜᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶜᶠᵃ, irange, jrange)
    Δxᶜᶠᵃ = Reactant.to_rarray(map(FT, Δxᶜᶠᵃ); sharding)
    Δxᶠᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δxᶠᶠᵃ, irange, jrange)
    Δxᶠᶠᵃ = Reactant.to_rarray(map(FT, Δxᶠᶠᵃ); sharding)
    Δyᶜᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶜᶜᵃ, irange, jrange)
    Δyᶜᶜᵃ = Reactant.to_rarray(map(FT, Δyᶜᶜᵃ); sharding)
    Δyᶠᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶠᶜᵃ, irange, jrange)
    Δyᶠᶜᵃ = Reactant.to_rarray(map(FT, Δyᶠᶜᵃ); sharding)
    Δyᶜᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶜᶠᵃ, irange, jrange)
    Δyᶜᶠᵃ = Reactant.to_rarray(map(FT, Δyᶜᶠᵃ); sharding)
    Δyᶠᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Δyᶠᶠᵃ, irange, jrange)
    Δyᶠᶠᵃ = Reactant.to_rarray(map(FT, Δyᶠᶠᵃ); sharding)
    Azᶜᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶜᶜᵃ, irange, jrange)
    Azᶜᶜᵃ = Reactant.to_rarray(map(FT, Azᶜᶜᵃ); sharding)
    Azᶠᶜᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶠᶜᵃ, irange, jrange)
    Azᶠᶜᵃ = Reactant.to_rarray(map(FT, Azᶠᶜᵃ); sharding)
    Azᶜᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶜᶠᵃ, irange, jrange)
    Azᶜᶠᵃ = Reactant.to_rarray(map(FT, Azᶜᶠᵃ); sharding)
    Azᶠᶠᵃ = Oceananigans.OrthogonalSphericalShellGrids.partition_tripolar_metric(global_grid, :Azᶠᶠᵃ, irange, jrange)
    Azᶠᶠᵃ = Reactant.to_rarray(map(FT, Azᶠᶠᵃ); sharding)

    grid = OrthogonalSphericalShellGrid{Periodic,RightConnected,Bounded}(arch,
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, global_grid.Lz),
        λᶜᶜᵃ,
        λᶠᶜᵃ,
        λᶜᶠᵃ,
        λᶠᶠᵃ,
        φᶜᶜᵃ,
        φᶠᶜᵃ,
        φᶜᶠᵃ,
        φᶠᶠᵃ,
        on_architecture(arch, global_grid.z),
        Δxᶜᶜᵃ,
        Δxᶠᶜᵃ,
        Δxᶜᶠᵃ,
        Δxᶠᶠᵃ,
        Δyᶜᶜᵃ,
        Δyᶠᶜᵃ,
        Δyᶜᶠᵃ,
        Δyᶠᶠᵃ,
        Azᶜᶜᵃ,
        Azᶠᶜᵃ,
        Azᶜᶠᵃ,
        Azᶠᶠᵃ,
        convert(FT, global_grid.radius),
        global_grid.conformal_mapping)

    return grid
end

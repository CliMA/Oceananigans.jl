using Oceananigans.BoundaryConditions: GWNFBC, has_target_transport, get_target_transport, TransportTimeSeries
using Oceananigans.DistributedComputations: all_reduce, device_all_reduce!, synchronize_communication!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: column_depthᶜᶜᵃ
using Oceananigans.Operators: Azᶜᶜᶜ

#####
##### Targeted barotropic transports through GravityWaveRadiation boundaries
#####

# Multi-region fields carry a `MultiRegionObject` of conditions, which cannot hold a target.
side_condition(bcs::FieldBoundaryConditions, side) = getproperty(bcs, side)
side_condition(bcs, side) = nothing

targeted_side(bc) = false
targeted_side(bc::GWNFBC) = has_target_transport(bc.classification.scheme)

has_targeted_barotropic_sides(U_bcs, V_bcs) =
    targeted_side(side_condition(U_bcs, :west))  || targeted_side(side_condition(U_bcs, :east)) ||
    targeted_side(side_condition(V_bcs, :south)) || targeted_side(side_condition(V_bcs, :north))

# The target of each side (or `nothing`); the same targets pin `(U, V)` during the substeps and `(Ũ, Ṽ)` after them.
function materialize_barotropic_boundary_transport(U, V, grid, export_forcing = false)
    targets = (west  = side_transport(side_condition(U.boundary_conditions, :west),  grid),
               east  = side_transport(side_condition(U.boundary_conditions, :east),  grid),
               south = side_transport(side_condition(V.boundary_conditions, :south), grid),
               north = side_transport(side_condition(V.boundary_conditions, :north), grid))
    return materialize_barotropic_boundary_transport(targets, U, grid, Val(exports_free_surface_forcing(export_forcing)))
end

materialize_barotropic_boundary_transport(targets, U, grid, ::Val{false}) = materialize_barotropic_boundary_transport(targets, grid)

function materialize_barotropic_boundary_transport(targets, U, grid, ::Val{true})
    lengths = targeted_wet_lengths(targets, U, grid)
    total_length = sum(lengths)

    total_length > 0 ||
        throw(ArgumentError("export_free_surface_forcing = true needs at least one wet side with a `target_transport`."))

    exported_volume_flux = Ref(zero(eltype(grid)))
    outward_shares = (west = - lengths.west / total_length, east  = lengths.east  / total_length,
                      south = - lengths.south / total_length, north = lengths.north / total_length)

    exporting_targets = map((target, share) -> exporting_target(target, share, exported_volume_flux), targets, outward_shares)
    boundary_transport = materialize_barotropic_boundary_transport(exporting_targets, grid)
    volume_flux = Field{Center, Center, Nothing}(grid)

    return FreeSurfaceForcingExport(boundary_transport, volume_flux, exported_volume_flux)
end

materialize_barotropic_boundary_transport(targets, grid) = all(isnothing, targets) ? nothing : targets

# Only the ranks on a physical boundary hold its target, but every rank joins the reductions of a targeted side.
function materialize_barotropic_boundary_transport(targets, grid::DistributedGrid)
    targeted = all_reduce(+, [Int(!isnothing(target)) for target in targets], architecture(grid))
    all(iszero, targeted) && return nothing
    sums = on_architecture(architecture(grid), zeros(eltype(grid), 8))
    return DistributedBarotropicTargets(targets, sums)
end

side_transport(bc, grid) = nothing
side_transport(bc::GWNFBC, grid) =
    has_target_transport(bc.classification.scheme) ? materialize_target(eltype(grid), get_target_transport(bc.classification.scheme, grid)) : nothing

materialize_target(FT, target::Number) = convert(FT, target)
materialize_target(FT, target) = target

current_target(FT, ::Nothing, time) = nothing
current_target(FT, target::Number, time) = convert(FT, target)
current_target(FT, target::TransportTimeSeries, time) = convert(FT, target(time))
current_target(FT, target, time) = convert(FT, target(time)) # any other target is a callable of the model time

#####
##### Exporting the free surface forcing through the targeted sides
#####

exports_free_surface_forcing(export_forcing::Bool) = export_forcing
exports_free_surface_forcing(boundary_transport) = false

"""
$(TYPEDEF)

Barotropic targets whose targeted sides also carry the area-integrated free surface forcing `∫Fη dA` out of the
domain, split in proportion to their wet lengths, so that the forcing leaves the domain volume unchanged.

$(TYPEDFIELDS)
"""
struct FreeSurfaceForcingExport{B, F, R}
    "the barotropic targets, with each targeted side shifted by its share of `exported_volume_flux`"
    boundary_transport :: B
    "the free surface forcing times the cell area of each wet column [m³ s⁻¹]"
    volume_flux :: F
    "the free surface forcing integrated over the wet surface [m³ s⁻¹], updated every time step"
    exported_volume_flux :: R
end

exports_free_surface_forcing(::FreeSurfaceForcingExport) = true

Adapt.adapt_structure(to, ::FreeSurfaceForcingExport) = nothing

"""
$(TYPEDEF)

The target of one side plus its `outward_share` of the exported volume flux, positive out of the domain.

$(TYPEDFIELDS)
"""
struct ExportingTarget{T, FT, R}
    "the target transport of the side"
    target :: T
    "the side's wet length over the total targeted wet length, signed so that it points out of the domain"
    outward_share :: FT
    "the free surface forcing integrated over the wet surface [m³ s⁻¹]"
    exported_volume_flux :: R
end

exporting_target(::Nothing, outward_share, exported_volume_flux) = nothing
exporting_target(target, outward_share, exported_volume_flux) = ExportingTarget(target, outward_share, exported_volume_flux)

current_target(FT, target::ExportingTarget, time) =
    current_target(FT, target.target, time) + convert(FT, target.outward_share * target.exported_volume_flux[])

# The wet length of each targeted side, summed over all ranks
function targeted_wet_lengths(targets, U, grid)
    arch = architecture(grid)
    sums = on_architecture(arch, zeros(eltype(grid), 8))
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    integrate! = _integrate_barotropic_face!(device(arch), workgroup, workgroup)

    integrate_wet_length!(integrate!, sums, 1, targets.west,  U, grid, 1,           Val(:x))
    integrate_wet_length!(integrate!, sums, 3, targets.east,  U, grid, grid.Nx + 1, Val(:x))
    integrate_wet_length!(integrate!, sums, 5, targets.south, U, grid, 1,           Val(:y))
    integrate_wet_length!(integrate!, sums, 7, targets.north, U, grid, grid.Ny + 1, Val(:y))

    sums = all_reduce(+, on_architecture(CPU(), sums), arch)

    return (west = sums[2], east = sums[4], south = sums[6], north = sums[8])
end

integrate_wet_length!(integrate!, sums, n, ::Nothing, U, grid, index, direction) = nothing
integrate_wet_length!(integrate!, sums, n, target, U, grid, index, direction) = integrate!(sums, n, U, grid, index, direction)

@kernel function _compute_free_surface_volume_flux!(volume_flux, grid, F, clock, model_fields)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1
    wet = column_depthᶜᶜᵃ(i, j, grid) > 0
    @inbounds volume_flux[i, j, 1] = ifelse(wet, F(i, j, k_top, grid, clock, model_fields) * Azᶜᶜᶜ(i, j, 1, grid), zero(grid))
end

compute_exported_volume_flux!(boundary_transport, F, clock, model_fields) = nothing

function compute_exported_volume_flux!(forcing_export::FreeSurfaceForcingExport, F, clock, model_fields)
    volume_flux = forcing_export.volume_flux
    grid = volume_flux.grid
    launch!(architecture(grid), grid, :xy, _compute_free_surface_volume_flux!, volume_flux, grid, F, clock, model_fields)
    forcing_export.exported_volume_flux[] = sum(volume_flux)
    return nothing
end

"""
$(TYPEDEF)

Targets of a distributed grid: the local `targets` (`nothing` on sides that are not a physical boundary of this rank)
and a device buffer for the face integrals and wet lengths of the four sides, summed over all ranks every pin.

$(TYPEDFIELDS)
"""
struct DistributedBarotropicTargets{T, S}
    targets :: T
    sums    :: S
end

Adapt.adapt_structure(to, ::DistributedBarotropicTargets) = nothing

#####
##### Pinning a face: one pre-configured launch per targeted side and substep
#####

const FACE_WORKGROUP_SIZE = 256

@inline face_length(grid, ::Val{:x}) = grid.Ny
@inline face_length(grid, ::Val{:y}) = grid.Nx

@inline face_column(n, index, ::Val{:x}) = (index, n)
@inline face_column(n, index, ::Val{:y}) = (n, index)

# Width of a column along the face, zero where the column is dry
@inline wet_face_width(i, j, grid, ::Val{:x}) = ifelse(column_depthᶠᶜᵃ(i, j, grid) > 0, Δyᶠᶜᵃ(i, j, 1, grid), zero(grid))
@inline wet_face_width(i, j, grid, ::Val{:y}) = ifelse(column_depthᶜᶠᵃ(i, j, grid) > 0, Δxᶜᶠᵃ(i, j, 1, grid), zero(grid))

# A single workgroup strides along the face: the partial sums of the transport and of the wet length go through local
# memory, work item 1 adds them in a fixed order (so the shift is deterministic) and every wet column is shifted by the
# same amount, which lands the face integral exactly on the target.
@kernel function _pin_barotropic_face!(U, grid, index, target, direction)
    t = @index(Local, Linear)
    Σ∫U = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)
    ΣL  = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)

    ∫U = zero(grid)
    L  = zero(grid)
    for n in t:FACE_WORKGROUP_SIZE:face_length(grid, direction)
        i, j = face_column(n, index, direction)
        Δl = wet_face_width(i, j, grid, direction)
        @inbounds ∫U += U[i, j, 1] * Δl
        L += Δl
    end
    @inbounds Σ∫U[t] = ∫U
    @inbounds ΣL[t]  = L
    @synchronize

    if t == 1
        ∫U = zero(grid)
        L  = zero(grid)
        for m in 1:FACE_WORKGROUP_SIZE
            @inbounds ∫U += Σ∫U[m]
            @inbounds L  += ΣL[m]
        end
        @inbounds Σ∫U[1] = ifelse(L > 0, (∫U - target) / L, zero(grid))
    end
    @synchronize

    @inbounds shift = Σ∫U[1]
    for n in t:FACE_WORKGROUP_SIZE:face_length(grid, direction)
        i, j = face_column(n, index, direction)
        wet = wet_face_width(i, j, grid, direction) > 0
        @inbounds U[i, j, 1] -= ifelse(wet, shift, zero(shift))
    end
end

# The face integral and wet length of one side of the local grid, from a single workgroup, into `sums[n:n+1]`
@kernel function _integrate_barotropic_face!(sums, n, U, grid, index, direction)
    t = @index(Local, Linear)
    Σ∫U = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)
    ΣL  = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)

    ∫U = zero(grid)
    L  = zero(grid)
    for m in t:FACE_WORKGROUP_SIZE:face_length(grid, direction)
        i, j = face_column(m, index, direction)
        Δl = wet_face_width(i, j, grid, direction)
        @inbounds ∫U += U[i, j, 1] * Δl
        L += Δl
    end
    @inbounds Σ∫U[t] = ∫U
    @inbounds ΣL[t]  = L
    @synchronize

    if t == 1
        ∫U = zero(grid)
        L  = zero(grid)
        for m in 1:FACE_WORKGROUP_SIZE
            @inbounds ∫U += Σ∫U[m]
            @inbounds L  += ΣL[m]
        end
        @inbounds sums[n]     = ∫U
        @inbounds sums[n + 1] = L
    end
end

# Shifts the face columns `first:last` along the face, which include the halo columns shared with neighbouring ranks,
# by the shift of the globally summed integral and wet length in `sums[n:n+1]`
@kernel function _shift_barotropic_face!(U, grid, index, sums, n, target, direction, first, last)
    t = @index(Local, Linear)
    @inbounds ∫U = sums[n]
    @inbounds L  = sums[n + 1]
    shift = ifelse(L > 0, (∫U - target) / L, zero(grid))
    for m in first+t-1:FACE_WORKGROUP_SIZE:last
        i, j = face_column(m, index, direction)
        wet = wet_face_width(i, j, grid, direction) > 0
        @inbounds U[i, j, 1] -= ifelse(wet, shift, zero(shift))
    end
end

configure_face_pin(arch, grid, field, index, ::Nothing, direction) = nothing

# Built once per barotropic step, like the other substep kernels: a single static workgroup and device-converted arguments
function configure_face_pin(arch, grid, field, index, target, direction)
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    kernel! = _pin_barotropic_face!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (field, grid, index, target, direction))
    return (; kernel!, args)
end

configure_face_pins(arch, grid, U, V, ::Nothing, time) = nothing
configure_face_pins(arch, grid, U, V, forcing_export::FreeSurfaceForcingExport, time) =
    configure_face_pins(arch, grid, U, V, forcing_export.boundary_transport, time)

function configure_face_pins(arch, grid, U, V, targets, time)
    FT = eltype(grid)
    return (west  = configure_face_pin(arch, grid, U, 1,           current_target(FT, targets.west,  time), Val(:x)),
            east  = configure_face_pin(arch, grid, U, grid.Nx + 1, current_target(FT, targets.east,  time), Val(:x)),
            south = configure_face_pin(arch, grid, V, 1,           current_target(FT, targets.south, time), Val(:y)),
            north = configure_face_pin(arch, grid, V, grid.Ny + 1, current_target(FT, targets.north, time), Val(:y)))
end

@inline pin_barotropic_face!(::Nothing) = nothing
@inline pin_barotropic_face!(pin) = pin.kernel!(pin.args...)

@inline pin_barotropic_faces!(::Nothing) = nothing

@inline function pin_barotropic_faces!(pins)
    pin_barotropic_face!(pins.west)
    pin_barotropic_face!(pins.east)
    pin_barotropic_face!(pins.south)
    pin_barotropic_face!(pins.north)
    return nothing
end

#####
##### Distributed grids: local face integrals, one all-reduce, then a uniform shift on the ranks that own the side
#####

struct DistributedFacePins{A, I, S, T, D}
    architecture :: A
    integrate!   :: I
    shift!       :: S
    sides        :: T # per side: (device-converted field, grid, index, direction, target), or `nothing`
    sums         :: D
end

distributed_side(arch, grid, field, index, ::Nothing, direction) = nothing
distributed_side(arch, grid, field, index, target, direction) =
    (args = convert_to_device(arch, (field, grid, index)), direction, target, range = shifted_range(arch, grid, direction))

# The neighbouring ranks along the face hold the same global shift, so the halo columns they share are shifted too
function shifted_range(arch, grid, ::Val{:x})
    first = isnothing(arch.connectivity.south) ? 1 : 1 - grid.Hy
    last  = isnothing(arch.connectivity.north) ? grid.Ny : grid.Ny + grid.Hy
    return (first, last)
end

function shifted_range(arch, grid, ::Val{:y})
    first = isnothing(arch.connectivity.west) ? 1 : 1 - grid.Hx
    last  = isnothing(arch.connectivity.east) ? grid.Nx : grid.Nx + grid.Hx
    return (first, last)
end

function configure_face_pins(arch, grid, U, V, targets::DistributedBarotropicTargets, time)
    FT = eltype(grid)
    local_targets = targets.targets
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    integrate! = _integrate_barotropic_face!(device(arch), workgroup, workgroup)
    shift!     = _shift_barotropic_face!(device(arch), workgroup, workgroup)

    sides = (west  = distributed_side(arch, grid, U, 1,           current_target(FT, local_targets.west,  time), Val(:x)),
             east  = distributed_side(arch, grid, U, grid.Nx + 1, current_target(FT, local_targets.east,  time), Val(:x)),
             south = distributed_side(arch, grid, V, 1,           current_target(FT, local_targets.south, time), Val(:y)),
             north = distributed_side(arch, grid, V, grid.Ny + 1, current_target(FT, local_targets.north, time), Val(:y)))

    return DistributedFacePins(arch, integrate!, shift!, sides, targets.sums)
end

integrate_face!(pins, ::Nothing, n) = nothing
integrate_face!(pins, side, n) = pins.integrate!(pins.sums, n, side.args..., side.direction)

shift_face!(pins, ::Nothing, n) = nothing
shift_face!(pins, side, n) = pins.shift!(side.args..., pins.sums, n, side.target, side.direction, side.range...)

function pin_barotropic_faces!(pins::DistributedFacePins)
    fill!(pins.sums, 0)
    integrate_face!(pins, pins.sides.west,  1)
    integrate_face!(pins, pins.sides.east,  3)
    integrate_face!(pins, pins.sides.south, 5)
    integrate_face!(pins, pins.sides.north, 7)
    device_all_reduce!(+, pins.sums, pins.architecture)
    shift_face!(pins, pins.sides.west,  1)
    shift_face!(pins, pins.sides.east,  3)
    shift_face!(pins, pins.sides.south, 5)
    shift_face!(pins, pins.sides.north, 7)
    return nothing
end

enforce_barotropic_transport_targets!(arch, grid, U, V, ::Nothing, time) = nothing

function enforce_barotropic_transport_targets!(arch, grid, U, V, targets, time)
    # A pending asynchronous exchange would overwrite the shifted halo columns along the face
    synchronize_communication!((U, V))
    pin_barotropic_faces!(configure_face_pins(arch, grid, U, V, targets, time))
    return nothing
end

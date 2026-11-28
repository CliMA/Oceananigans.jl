using Oceananigans.BoundaryConditions: GWNFBC, NetInflow, get_target_transport, has_target_transport, net_inflow_value
using Oceananigans.DistributedComputations: Distributed, all_reduce, all_reduce!, synchronize_communication!

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

"""
$(TYPEDEF)

The `target_transport` of each side (`nothing` on the sides without one, and on distributed grids on the sides that are
not a physical boundary of this rank), and a device buffer for the face integrals and weights of the four sides
(`sums[1:8]`) and of the sides that share a `NetInflow` (`sums[9:10]`), summed over all ranks on distributed grids.

$(TYPEDFIELDS)
"""
struct BarotropicTargets{T, S}
    targets :: T
    sums    :: S
end

Adapt.adapt_structure(to, ::BarotropicTargets) = nothing

# The same targets pin `(U, V)` during the substeps and `(Ũ, Ṽ)` after them.
function materialize_barotropic_boundary_transport(U, V, grid)
    targets = (west  = side_transport(side_condition(U.boundary_conditions, :west),  grid),
               east  = side_transport(side_condition(U.boundary_conditions, :east),  grid),
               south = side_transport(side_condition(V.boundary_conditions, :south), grid),
               north = side_transport(side_condition(V.boundary_conditions, :north), grid))

    # Only the ranks on a physical boundary hold its target, but every rank joins the reductions of a targeted side
    targeted = all_reduce(+, count(!isnothing, targets), architecture(grid))
    targeted == 0 && return nothing
    sums = on_architecture(architecture(grid), zeros(eltype(grid), 10))

    return BarotropicTargets(targets, sums)
end

side_transport(bc, grid) = nothing
side_transport(bc::GWNFBC, grid) =
    has_target_transport(bc.classification.scheme) ? materialize_target(eltype(grid), get_target_transport(bc.classification.scheme, grid)) : nothing

materialize_target(FT, target::Number) = convert(FT, target)
materialize_target(FT, target) = target

current_target(FT, ::Nothing, time) = nothing
current_target(FT, target::Number, time) = convert(FT, target)
current_target(FT, target::NetInflow, time) = target
current_target(FT, target, time) = convert(FT, target(time)) # any other target is a callable of the model time

#####
##### Pinning a face: the face integral of each targeted side, one sum over the ranks, then a shift of its columns
#####

@inline face_length(grid, ::Val{:x}) = size(grid, 2)
@inline face_length(grid, ::Val{:y}) = size(grid, 1)

@inline face_column(n, index, ::Val{:x}) = (index, n)
@inline face_column(n, index, ::Val{:y}) = (n, index)

@inline face_depth(i, j, grid, ::Val{:x}) = column_depthᶠᶜᵃ(i, j, grid)
@inline face_depth(i, j, grid, ::Val{:y}) = column_depthᶜᶠᵃ(i, j, grid)

# Width of a column along the face, zero where the column is dry
@inline wet_face_width(i, j, grid, ::Val{:x}) = ifelse(face_depth(i, j, grid, Val(:x)) > 0, Δyᶠᶜᵃ(i, j, 1, grid), zero(grid))
@inline wet_face_width(i, j, grid, ::Val{:y}) = ifelse(face_depth(i, j, grid, Val(:y)) > 0, Δxᶜᶠᵃ(i, j, 1, grid), zero(grid))

# Weight Hᵖ of a column in the shift, zero where the column is dry
@inline depth_weight(i, j, grid, p, direction) =
    ifelse(face_depth(i, j, grid, direction) > 0, face_depth(i, j, grid, direction)^p, zero(grid))

# A single workgroup strides along one side of the local grid and adds the inflow sign s times the face integral of U,
# and the integral of Hᵖ, into `sums[n:n+1]`: work item 1 adds the partial sums in a fixed order, so the shift is
# deterministic.
@kernel function _integrate_barotropic_face!(sums, n, U, grid, index, direction, s, p)
    t = @index(Local, Linear)
    workgroup_size = @uniform prod(@groupsize())
    Σ∫U = @localmem eltype(grid) (workgroup_size,)
    ΣL  = @localmem eltype(grid) (workgroup_size,)

    ∫U = zero(grid)
    L  = zero(grid)
    for m in t:workgroup_size:face_length(grid, direction)
        i, j = face_column(m, index, direction)
        Δl = wet_face_width(i, j, grid, direction)
        @inbounds ∫U += s * U[i, j, 1] * Δl
        L += Δl * depth_weight(i, j, grid, p, direction)
    end
    @inbounds Σ∫U[t] = ∫U
    @inbounds ΣL[t]  = L
    @synchronize

    if t == 1
        ∫U = zero(grid)
        L  = zero(grid)
        for m in 1:workgroup_size
            @inbounds ∫U += Σ∫U[m]
            @inbounds L  += ΣL[m]
        end
        @inbounds sums[n]     += ∫U
        @inbounds sums[n + 1] += L
    end
end

# Each wet column `first:last` along the face, which include the halo columns shared with neighbouring ranks, moves by
# -s Hᵖ (∫U - target) / L, which lands the summed face integral ∫U in `sums[n]` on the target
@kernel function _shift_barotropic_face!(U, grid, index, sums, n, target, direction, s, p, first, last)
    t = @index(Local, Linear)
    workgroup_size = @uniform prod(@groupsize())
    @inbounds ∫U = sums[n]
    @inbounds L  = sums[n + 1]
    shift = ifelse(L > 0, (∫U - target) / L, zero(grid))
    for m in first+t-1:workgroup_size:last
        i, j = face_column(m, index, direction)
        @inbounds U[i, j, 1] -= s * shift * depth_weight(i, j, grid, p, direction)
    end
end

struct BarotropicFacePins{A, I, S, P, D}
    architecture :: A
    integrate!   :: I
    shift!       :: S
    sides        :: P # per side: device-converted (field, grid, index), direction, sign, exponent, slot, target and range
    sums         :: D
end

# A side pins its own transport into `sums[n:n+1]`; the sides that share a `NetInflow` pin their summed inflow into
# `sums[9:10]`, each with its inflow sign s
face_pin(arch, grid, field, index, ::Nothing, direction, s, n) = nothing
face_pin(arch, grid, field, index, target, direction, s, n) = face_pin(arch, grid, field, index, direction, 1, 0, n, target)

face_pin(arch, grid, field, index, target::NetInflow, direction, s, n) =
    face_pin(arch, grid, field, index, direction, s, convert(eltype(grid), target.depth_exponent), 9,
             convert(eltype(grid), net_inflow_value(target.transport)))

face_pin(arch, grid, field, index, direction, sign, p, slot, target) =
    (args = convert_to_device(arch, (field, grid, index)), direction, sign, p, slot, target,
     range = shifted_range(arch, grid, direction))

shifted_range(arch, grid, direction) = (1, face_length(grid, direction))

# The neighbouring ranks along the face hold the same global shift, so the halo columns they share are shifted too
function shifted_range(arch::Distributed, grid, ::Val{:x})
    first = isnothing(arch.connectivity.south) ? 1 : 1 - grid.Hy
    last  = isnothing(arch.connectivity.north) ? grid.Ny : grid.Ny + grid.Hy
    return (first, last)
end

function shifted_range(arch::Distributed, grid, ::Val{:y})
    first = isnothing(arch.connectivity.west) ? 1 : 1 - grid.Hx
    last  = isnothing(arch.connectivity.east) ? grid.Nx : grid.Nx + grid.Hx
    return (first, last)
end

configure_face_pins(arch, grid, U, V, ::Nothing, time) = nothing

# Built once per barotropic step, like the other substep kernels: a single static workgroup of 256 work items, which the
# kernels read with `@groupsize`, and device-converted arguments
function configure_face_pins(arch, grid, U, V, targets::BarotropicTargets, time)
    FT = eltype(grid)
    local_targets = targets.targets
    workgroup = StaticSize((256,))
    integrate! = _integrate_barotropic_face!(device(arch), workgroup, workgroup)
    shift!     = _shift_barotropic_face!(device(arch), workgroup, workgroup)

    sides = (west  = face_pin(arch, grid, U, 1,                 current_target(FT, local_targets.west,  time), Val(:x),  1, 1),
             east  = face_pin(arch, grid, U, size(grid, 1) + 1, current_target(FT, local_targets.east,  time), Val(:x), -1, 3),
             south = face_pin(arch, grid, V, 1,                 current_target(FT, local_targets.south, time), Val(:y),  1, 5),
             north = face_pin(arch, grid, V, size(grid, 2) + 1, current_target(FT, local_targets.north, time), Val(:y), -1, 7))

    return BarotropicFacePins(arch, integrate!, shift!, sides, targets.sums)
end

integrate_face!(pins, ::Nothing) = nothing
integrate_face!(pins, side) = pins.integrate!(pins.sums, side.slot, side.args..., side.direction, side.sign, side.p)

shift_face!(pins, ::Nothing) = nothing
shift_face!(pins, side) =
    pins.shift!(side.args..., pins.sums, side.slot, side.target, side.direction, side.sign, side.p, side.range...)

pin_barotropic_faces!(::Nothing) = nothing

function pin_barotropic_faces!(pins::BarotropicFacePins)
    fill!(pins.sums, 0)
    foreach(side -> integrate_face!(pins, side), pins.sides)
    all_reduce!(+, pins.sums, pins.architecture)
    foreach(side -> shift_face!(pins, side), pins.sides)
    return nothing
end

enforce_barotropic_transport_targets!(arch, grid, U, V, ::Nothing, time) = nothing

function enforce_barotropic_transport_targets!(arch, grid, U, V, targets, time)
    # A pending asynchronous exchange would overwrite the shifted halo columns along the face
    synchronize_communication!((U, V))
    pin_barotropic_faces!(configure_face_pins(arch, grid, U, V, targets, time))
    return nothing
end

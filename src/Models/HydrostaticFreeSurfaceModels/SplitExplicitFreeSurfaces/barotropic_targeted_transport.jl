using Oceananigans.BoundaryConditions: GWNFBC, has_target_transport, get_target_transport

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

# TODO: add support for distributed grids
function validate_free_surface_boundary_conditions(::SplitExplicitFreeSurface, boundary_conditions, grid)
    targeted = has_targeted_barotropic_sides(get(boundary_conditions, :U, nothing), get(boundary_conditions, :V, nothing))
    if targeted && grid isa DistributedGrid
        throw(ArgumentError("`target_transport` on `GravityWaveRadiation` boundary conditions is not supported on distributed grids."))
    end
    return nothing
end

# The target of each side (or `nothing`); the same targets pin `(U, V)` during the substeps and `(Ũ, Ṽ)` after them.
function materialize_barotropic_boundary_transport(U, V, grid)
    has_targeted_barotropic_sides(U.boundary_conditions, V.boundary_conditions) || return nothing
    return (west  = side_transport(side_condition(U.boundary_conditions, :west),  grid),
            east  = side_transport(side_condition(U.boundary_conditions, :east),  grid),
            south = side_transport(side_condition(V.boundary_conditions, :south), grid),
            north = side_transport(side_condition(V.boundary_conditions, :north), grid))
end

side_transport(bc, grid) = nothing
side_transport(bc::GWNFBC, grid) =
    has_target_transport(bc.classification.scheme) ? convert(eltype(grid), get_target_transport(bc.classification.scheme, grid)) : nothing

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

configure_face_pin(arch, grid, field, index, ::Nothing, direction) = nothing

# Built once per barotropic step, like the other substep kernels: a single static workgroup and device-converted arguments
function configure_face_pin(arch, grid, field, index, target, direction)
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    kernel! = _pin_barotropic_face!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (field, grid, index, target, direction))
    return (; kernel!, args)
end

configure_face_pins(arch, grid, U, V, ::Nothing) = nothing
configure_face_pins(arch, grid, U, V, targets) =
    (west  = configure_face_pin(arch, grid, U, 1,           targets.west,  Val(:x)),
     east  = configure_face_pin(arch, grid, U, grid.Nx + 1, targets.east,  Val(:x)),
     south = configure_face_pin(arch, grid, V, 1,           targets.south, Val(:y)),
     north = configure_face_pin(arch, grid, V, grid.Ny + 1, targets.north, Val(:y)))

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

enforce_barotropic_transport_targets!(arch, grid, U, V, targets) = pin_barotropic_faces!(configure_face_pins(arch, grid, U, V, targets))

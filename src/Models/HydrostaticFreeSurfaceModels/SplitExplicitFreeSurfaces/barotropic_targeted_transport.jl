using Oceananigans.BoundaryConditions: GWNFBC, NetInflow, get_target_transport, has_target_transport, net_inflow_value

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
function HydrostaticFreeSurfaceModels.validate_free_surface_boundary_conditions(::SplitExplicitFreeSurface, boundary_conditions, grid)
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
    has_target_transport(bc.classification.scheme) ? side_target(get_target_transport(bc.classification.scheme, grid), grid) : nothing

side_target(target, grid) = convert(eltype(grid), target)
side_target(target::NetInflow, grid) = target

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
configure_face_pin(arch, grid, field, index, ::NetInflow, direction) = nothing

# Built once per barotropic step, like the other substep kernels: a single static workgroup and device-converted arguments
function configure_face_pin(arch, grid, field, index, target, direction)
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    kernel! = _pin_barotropic_face!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (field, grid, index, target, direction))
    return (; kernel!, args)
end

#####
##### Pinning the net inflow of the sides that share a `NetInflow`
#####

# Inflow sign of each side (west, east, south, north), zero for the sides that do not share the net inflow
pooled(target::NetInflow, sign) = sign
pooled(target, sign) = 0
net_inflow_weights(targets) = map(pooled, (targets.west, targets.east, targets.south, targets.north), (1, -1, 1, -1))

net_inflow_target(targets, grid) =
    convert(eltype(grid), net_inflow_value(first(target.transport for target in values(targets) if target isa NetInflow)))

# As `_pin_barotropic_face!`, over every pooled face at once: with sₙ = ±1 the inflow sign of side n,
# Q = Σₙ sₙ ∮ U dl and L = Σₙ ∮ dl over the wet columns, every wet column of side n is shifted by -sₙ (Q - target) / L.
@kernel function _pin_net_barotropic_inflow!(U, V, grid, target, weights)
    t = @index(Local, Linear)
    ΣQ = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)
    ΣL = @localmem eltype(grid) (FACE_WORKGROUP_SIZE,)
    sʷ, sᵉ, sˢ, sⁿ = weights
    Nx, Ny = grid.Nx, grid.Ny

    Q = zero(grid)
    L = zero(grid)
    for n in t:FACE_WORKGROUP_SIZE:Ny
        Δlʷ = abs(sʷ) * wet_face_width(1, n, grid, Val(:x))
        Δlᵉ = abs(sᵉ) * wet_face_width(Nx + 1, n, grid, Val(:x))
        @inbounds Q += sʷ * U[1, n, 1] * Δlʷ + sᵉ * U[Nx + 1, n, 1] * Δlᵉ
        L += Δlʷ + Δlᵉ
    end
    for n in t:FACE_WORKGROUP_SIZE:Nx
        Δlˢ = abs(sˢ) * wet_face_width(n, 1, grid, Val(:y))
        Δlⁿ = abs(sⁿ) * wet_face_width(n, Ny + 1, grid, Val(:y))
        @inbounds Q += sˢ * V[n, 1, 1] * Δlˢ + sⁿ * V[n, Ny + 1, 1] * Δlⁿ
        L += Δlˢ + Δlⁿ
    end
    @inbounds ΣQ[t] = Q
    @inbounds ΣL[t] = L
    @synchronize

    if t == 1
        Q = zero(grid)
        L = zero(grid)
        for m in 1:FACE_WORKGROUP_SIZE
            @inbounds Q += ΣQ[m]
            @inbounds L += ΣL[m]
        end
        @inbounds ΣQ[1] = ifelse(L > 0, (Q - target) / L, zero(grid))
    end
    @synchronize

    @inbounds shift = ΣQ[1]
    for n in t:FACE_WORKGROUP_SIZE:Ny
        @inbounds U[1, n, 1]      -= ifelse(wet_face_width(1, n, grid, Val(:x)) > 0, sʷ * shift, zero(shift))
        @inbounds U[Nx + 1, n, 1] -= ifelse(wet_face_width(Nx + 1, n, grid, Val(:x)) > 0, sᵉ * shift, zero(shift))
    end
    for n in t:FACE_WORKGROUP_SIZE:Nx
        @inbounds V[n, 1, 1]      -= ifelse(wet_face_width(n, 1, grid, Val(:y)) > 0, sˢ * shift, zero(shift))
        @inbounds V[n, Ny + 1, 1] -= ifelse(wet_face_width(n, Ny + 1, grid, Val(:y)) > 0, sⁿ * shift, zero(shift))
    end
end

function configure_net_inflow_pin(arch, grid, U, V, targets)
    weights = net_inflow_weights(targets)
    all(iszero, weights) && return nothing
    workgroup = StaticSize((FACE_WORKGROUP_SIZE,))
    kernel! = _pin_net_barotropic_inflow!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (U, V, grid, net_inflow_target(targets, grid), weights))
    return (; kernel!, args)
end

configure_face_pins(arch, grid, U, V, ::Nothing) = nothing
configure_face_pins(arch, grid, U, V, targets) =
    (west  = configure_face_pin(arch, grid, U, 1,           targets.west,  Val(:x)),
     east  = configure_face_pin(arch, grid, U, grid.Nx + 1, targets.east,  Val(:x)),
     south = configure_face_pin(arch, grid, V, 1,           targets.south, Val(:y)),
     north = configure_face_pin(arch, grid, V, grid.Ny + 1, targets.north, Val(:y)),
     net   = configure_net_inflow_pin(arch, grid, U, V, targets))

@inline pin_barotropic_face!(::Nothing) = nothing
@inline pin_barotropic_face!(pin) = pin.kernel!(pin.args...)

@inline pin_barotropic_faces!(::Nothing) = nothing

@inline function pin_barotropic_faces!(pins)
    pin_barotropic_face!(pins.west)
    pin_barotropic_face!(pins.east)
    pin_barotropic_face!(pins.south)
    pin_barotropic_face!(pins.north)
    pin_barotropic_face!(pins.net)
    return nothing
end

enforce_barotropic_transport_targets!(arch, grid, U, V, targets) = pin_barotropic_faces!(configure_face_pins(arch, grid, U, V, targets))

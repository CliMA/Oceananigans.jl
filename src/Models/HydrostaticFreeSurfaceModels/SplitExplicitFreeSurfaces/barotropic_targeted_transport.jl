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

@inline face_length(grid, ::Val{:x}) = size(grid, 2)
@inline face_length(grid, ::Val{:y}) = size(grid, 1)

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
    workgroup_size = @uniform prod(@groupsize())
    Σ∫U = @localmem eltype(grid) (workgroup_size,)
    ΣL  = @localmem eltype(grid) (workgroup_size,)

    ∫U = zero(grid)
    L  = zero(grid)
    for n in t:workgroup_size:face_length(grid, direction)
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
        for m in 1:workgroup_size
            @inbounds ∫U += Σ∫U[m]
            @inbounds L  += ΣL[m]
        end
        @inbounds Σ∫U[1] = ifelse(L > 0, (∫U - target) / L, zero(grid))
    end
    @synchronize

    @inbounds shift = Σ∫U[1]
    for n in t:workgroup_size:face_length(grid, direction)
        i, j = face_column(n, index, direction)
        wet = wet_face_width(i, j, grid, direction) > 0
        @inbounds U[i, j, 1] -= ifelse(wet, shift, zero(shift))
    end
end

configure_face_pin(arch, grid, field, index, ::Nothing,   direction) = nothing
configure_face_pin(arch, grid, field, index, ::NetInflow, direction) = nothing

# Built once per barotropic step, like the other substep kernels: a single static workgroup of 256 work items, which the
# kernels read with `@groupsize`, and device-converted arguments
function configure_face_pin(arch, grid, field, index, target, direction)
    workgroup = StaticSize((256,))
    kernel! = _pin_barotropic_face!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (field, grid, index, target, direction))
    return (; kernel!, args)
end

#####
##### Pinning the net inflow of the sides that share a `NetInflow`
#####

# Inflow sign of each side (west, east, south, north), zero for the sides that do not share the net inflow
net_inflow_sign(target::NetInflow, sign) = sign
net_inflow_sign(target, sign) = 0
net_inflow_signs(targets) = map(net_inflow_sign, (targets.west, targets.east, targets.south, targets.north), (1, -1, 1, -1))

function net_inflow_target(targets, grid)
    target = first(target for target in values(targets) if target isa NetInflow)
    return convert(eltype(grid), net_inflow_value(target.transport)), convert(eltype(grid), target.depth_exponent)
end

# Weight Hᵖ of a boundary column in the net-inflow correction, zero where the column is dry
@inline depth_weightᶠᶜᵃ(i, j, grid, p) = ifelse(column_depthᶠᶜᵃ(i, j, grid) > 0, column_depthᶠᶜᵃ(i, j, grid)^p, zero(grid))
@inline depth_weightᶜᶠᵃ(i, j, grid, p) = ifelse(column_depthᶜᶠᵃ(i, j, grid) > 0, column_depthᶜᶠᵃ(i, j, grid)^p, zero(grid))

@kernel function _pin_net_barotropic_inflow!(U, V, grid, target, p, sʷ, sᵉ, sˢ, sⁿ)
    t = @index(Local, Linear)
    workgroup_size = @uniform prod(@groupsize())
    ΣQ = @localmem eltype(grid) (workgroup_size,)
    ΣL = @localmem eltype(grid) (workgroup_size,)

    # Partial sums of the net inflow Q = Σ s ∫ U dl and of L = Σ ∫ Hᵖ dl, first over the west and east sides...
    Q = zero(grid)
    L = zero(grid)
    for j in t:workgroup_size:size(grid, 2)
        Δyʷ = abs(sʷ) * wet_face_width(1, j, grid, Val(:x))
        Δyᵉ = abs(sᵉ) * wet_face_width(size(grid, 1) + 1, j, grid, Val(:x))
        @inbounds Q += sʷ * U[1, j, 1] * Δyʷ + sᵉ * U[size(grid, 1) + 1, j, 1] * Δyᵉ
        L += Δyʷ * column_depthᶠᶜᵃ(1, j, grid)^p + Δyᵉ * column_depthᶠᶜᵃ(size(grid, 1) + 1, j, grid)^p
    end

    # ...then over the south and north sides
    for i in t:workgroup_size:size(grid, 1)
        Δxˢ = abs(sˢ) * wet_face_width(i, 1, grid, Val(:y))
        Δxⁿ = abs(sⁿ) * wet_face_width(i, size(grid, 2) + 1, grid, Val(:y))
        @inbounds Q += sˢ * V[i, 1, 1] * Δxˢ + sⁿ * V[i, size(grid, 2) + 1, 1] * Δxⁿ
        L += Δxˢ * column_depthᶜᶠᵃ(i, 1, grid)^p + Δxⁿ * column_depthᶜᶠᵃ(i, size(grid, 2) + 1, grid)^p
    end
    @inbounds ΣQ[t] = Q
    @inbounds ΣL[t] = L

    # Locals do not survive `@synchronize` on the CPU (KernelAbstractions splits the kernel there): pass through ΣQ
    @synchronize

    # Work item 1 adds the partial sums in a fixed order, so the shift per unit weight (Q - target) / L is deterministic
    if t == 1
        Q = zero(grid)
        L = zero(grid)
        for m in 1:workgroup_size
            @inbounds Q += ΣQ[m]
            @inbounds L += ΣL[m]
        end
        @inbounds ΣQ[1] = ifelse(L > 0, (Q - target) / L, zero(grid))
    end
    @synchronize

    # Each wet column of a side with inflow sign s moves by -s Hᵖ (Q - target) / L, which lands Q on the target
    @inbounds shift = ΣQ[1]
    for j in t:workgroup_size:size(grid, 2)
        @inbounds U[1, j, 1]                 -= sʷ * shift * depth_weightᶠᶜᵃ(1, j, grid, p)
        @inbounds U[size(grid, 1) + 1, j, 1] -= sᵉ * shift * depth_weightᶠᶜᵃ(size(grid, 1) + 1, j, grid, p)
    end
    for i in t:workgroup_size:size(grid, 1)
        @inbounds V[i, 1, 1]                 -= sˢ * shift * depth_weightᶜᶠᵃ(i, 1, grid, p)
        @inbounds V[i, size(grid, 2) + 1, 1] -= sⁿ * shift * depth_weightᶜᶠᵃ(i, size(grid, 2) + 1, grid, p)
    end
end

function configure_net_inflow_pin(arch, grid, U, V, targets)
    signs = net_inflow_signs(targets)
    all(iszero, signs) && return nothing
    workgroup = StaticSize((256,))
    kernel! = _pin_net_barotropic_inflow!(device(arch), workgroup, workgroup)
    args = convert_to_device(arch, (U, V, grid, net_inflow_target(targets, grid)..., signs...))
    return (; kernel!, args)
end

configure_face_pins(arch, grid, U, V, ::Nothing) = nothing
configure_face_pins(arch, grid, U, V, targets) =
    (west  = configure_face_pin(arch, grid, U, 1,           targets.west,  Val(:x)),
     east  = configure_face_pin(arch, grid, U, size(grid, 1) + 1, targets.east,  Val(:x)),
     south = configure_face_pin(arch, grid, V, 1,           targets.south, Val(:y)),
     north = configure_face_pin(arch, grid, V, size(grid, 2) + 1, targets.north, Val(:y)),
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

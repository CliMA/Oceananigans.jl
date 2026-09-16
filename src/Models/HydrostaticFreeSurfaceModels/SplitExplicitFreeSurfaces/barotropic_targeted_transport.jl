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

# Checked on the boundary conditions the user passed, so every rank reaches the same verdict. The face
# integrals below are rank-local, hence the restriction; multi-region grids add their own method.
function validate_free_surface_boundary_conditions(::SplitExplicitFreeSurface, boundary_conditions, grid)
    targeted = has_targeted_barotropic_sides(get(boundary_conditions, :U, nothing), get(boundary_conditions, :V, nothing))
    if targeted && grid isa DistributedGrid
        throw(ArgumentError("`target_transport` on `GravityWaveRadiation` boundary conditions is not supported on distributed grids."))
    end
    return nothing
end

# Each targeted side stores its target, the wet length of its face and a reduced `Field` with the face integral
# of the transport, recomputed every substep. One group pins `U` and `V`, the other the filtered `Ũ` and `Ṽ`.
function materialize_barotropic_boundary_transport(U, V, Ũ, Ṽ, grid)
    has_targeted_barotropic_sides(U.boundary_conditions, V.boundary_conditions) || return nothing
    return (; barotropic = side_transports(U, V, grid), filtered = side_transports(Ũ, Ṽ, grid))
end

barotropic_sides(::Nothing) = nothing
barotropic_sides(boundary_transport) = boundary_transport.barotropic

filtered_sides(::Nothing) = nothing
filtered_sides(boundary_transport) = boundary_transport.filtered

side_transports(U, V, grid) = (west  = side_transport(side_condition(U.boundary_conditions, :west),  U, grid, Val(:west)),
                               east  = side_transport(side_condition(U.boundary_conditions, :east),  U, grid, Val(:east)),
                               south = side_transport(side_condition(V.boundary_conditions, :south), V, grid, Val(:south)),
                               north = side_transport(side_condition(V.boundary_conditions, :north), V, grid, Val(:north)))

side_transport(bc, field, grid, side) = nothing

function side_transport(bc::GWNFBC, field, grid, side)
    has_target_transport(bc.classification.scheme) || return nothing
    wet_length = face_wet_length(field, grid, side)
    wet_length > 0 || return nothing # a fully dry side carries no transport
    target = get_target_transport(bc.classification.scheme, grid)
    return (; target, wet_length, integral = face_integral(field, grid, side))
end

# Reductions over immersed grids skip dry columns, so the integrals only see the wet part of the face
face_integral(U, grid, ::Val{:west})  = Field(Integral(view(U, 1, :, :), dims = 2))
face_integral(U, grid, ::Val{:east})  = Field(Integral(view(U, grid.Nx + 1, :, :), dims = 2))
face_integral(V, grid, ::Val{:south}) = Field(Integral(view(V, :, 1, :), dims = 1))
face_integral(V, grid, ::Val{:north}) = Field(Integral(view(V, :, grid.Ny + 1, :), dims = 1))

function face_wet_length(field, grid, side)
    LX, LY, LZ = location(field)
    ones = Field{LX, LY, LZ}(grid)
    set!(ones, 1)
    return @allowscalar compute!(face_integral(ones, grid, side))[]
end

# The face is shifted uniformly over its wet columns so that its integral matches the target
@kernel function _pin_barotropic_face!(U, grid, i, ∫U, target, wet_length, ::Val{:x})
    j = @index(Global, Linear)
    @inbounds shift = (∫U[i, 1, 1] - target) / wet_length
    wet = column_depthᶠᶜᵃ(i, j, grid) > 0
    @inbounds U[i, j, 1] -= ifelse(wet, shift, zero(shift))
end

@kernel function _pin_barotropic_face!(V, grid, j, ∫V, target, wet_length, ::Val{:y})
    i = @index(Global, Linear)
    @inbounds shift = (∫V[1, j, 1] - target) / wet_length
    wet = column_depthᶜᶠᵃ(i, j, grid) > 0
    @inbounds V[i, j, 1] -= ifelse(wet, shift, zero(shift))
end

face_parameters(grid, ::Val{:x}) = KernelParameters(1:grid.Ny)
face_parameters(grid, ::Val{:y}) = KernelParameters(1:grid.Nx)

pin_barotropic_face!(arch, grid, field, index, ::Nothing, direction) = nothing

function pin_barotropic_face!(arch, grid, field, index, side, direction)
    compute!(side.integral)
    launch!(arch, grid, face_parameters(grid, direction), _pin_barotropic_face!,
            field, grid, index, side.integral, side.target, side.wet_length, direction)
    return nothing
end

enforce_barotropic_transport_targets!(arch, grid, U, V, ::Nothing) = nothing

function enforce_barotropic_transport_targets!(arch, grid, U, V, sides)
    pin_barotropic_face!(arch, grid, U, 1,           sides.west,  Val(:x))
    pin_barotropic_face!(arch, grid, U, grid.Nx + 1, sides.east,  Val(:x))
    pin_barotropic_face!(arch, grid, V, 1,           sides.south, Val(:y))
    pin_barotropic_face!(arch, grid, V, grid.Ny + 1, sides.north, Val(:y))
    return nothing
end

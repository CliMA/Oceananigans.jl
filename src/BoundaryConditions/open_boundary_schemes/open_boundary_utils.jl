using Oceananigans: defaults, instantiated_location
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶜᶜᵃ, immersed_peripheral_node
using Oceananigans.Operators: Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ

#####
##### Open boundary schemes whose exterior state is prognostic: the time stepper caches, steps and initializes it, and the
##### halo fill leaves it in place. `φ⁻` holds the state at the base of a stage.
#####

abstract type AbstractRadiationScheme{FT} end

const RVBC  = BoundaryCondition{<:Value{<:AbstractRadiationScheme}}
const RNFBC = BoundaryCondition{<:NormalFlow{<:AbstractRadiationScheme}}
const RBC   = Union{RVBC, RNFBC}

# Location type aliases used to dispatch halo filling on the field's staggering.
const FAA = Tuple{Face, Any, Any}
const AFA = Tuple{Any, Face, Any}

#####
##### Geometry of a boundary point, the same for every side
#####

# Index along the boundary normal of the point `n` cells from the exterior state into the interior
@inline normal_index(::Union{West, South, Bottom}, ::Center, N, n) = n
@inline normal_index(::Union{West, South, Bottom}, ::Face,   N, n) = n + 1
@inline normal_index(::Union{East, North, Top},    ℓ,        N, n) = N + 1 - n

@inline boundary_point(side::Union{West, East},   loc, grid, n, j, k) = (normal_index(side, loc[1], grid.Nx, n), j, k)
@inline boundary_point(side::Union{South, North}, loc, grid, n, i, k) = (i, normal_index(side, loc[2], grid.Ny, n), k)
@inline boundary_point(side::Union{Bottom, Top},  loc, grid, n, i, j) = (i, j, normal_index(side, loc[3], grid.Nz, n))

# Boundary-normal velocity on the face `n` faces from the boundary into the interior
@inline normal_velocity(side::Union{West, East},   model_fields, grid, n, j, k) = @inbounds model_fields.u[normal_index(side, Face(), grid.Nx, n), j, k]
@inline normal_velocity(side::Union{South, North}, model_fields, grid, n, i, k) = @inbounds model_fields.v[i, normal_index(side, Face(), grid.Ny, n), k]
@inline normal_velocity(side::Union{Bottom, Top},  model_fields, grid, n, i, j) = @inbounds model_fields.w[i, j, normal_index(side, Face(), grid.Nz, n)]

# Reciprocal of the spacing between the exterior state and the first interior value
@inline reciprocal_boundary_spacing(side::Union{West, East},   grid, j, k) = Δx⁻¹ᶠᶜᶜ(normal_index(side, Face(), grid.Nx, 0), j, k, grid)
@inline reciprocal_boundary_spacing(side::Union{South, North}, grid, i, k) = Δy⁻¹ᶜᶠᶜ(i, normal_index(side, Face(), grid.Ny, 0), k, grid)
@inline reciprocal_boundary_spacing(side::Union{Bottom, Top},  grid, i, j) = Δz⁻¹ᶜᶜᶠ(i, j, normal_index(side, Face(), grid.Nz, 0), grid)

# Component of a boundary-normal velocity pointing out of the domain
@inline outward(::Union{West, South, Bottom}, U) = -U
@inline outward(::Union{East, North, Top},    U) =  U

# Whether the first interior cell is immersed
@inline immersed_boundary_point(side::Union{West, East},   loc, grid, j, k) = immersed_peripheral_node(normal_index(side, Center(), grid.Nx, 1), j, k, grid, Center(), loc[2], loc[3])
@inline immersed_boundary_point(side::Union{South, North}, loc, grid, i, k) = immersed_peripheral_node(i, normal_index(side, Center(), grid.Ny, 1), k, grid, loc[1], Center(), loc[3])
@inline immersed_boundary_point(side::Union{Bottom, Top},  loc, grid, i, j) = immersed_peripheral_node(i, j, normal_index(side, Center(), grid.Nz, 1), grid, loc[1], loc[2], Center())

# A boundary-normal velocity is advected by its own value `ψ`, a tracer or a tangential velocity by the normal flow
@inline advecting_velocity(::NFBC, side, model_fields, grid, n, i, j, ψ) = ψ
@inline advecting_velocity(bc, side, model_fields, grid, n, i, j, ψ) = normal_velocity(side, model_fields, grid, n, i, j)

#####
##### Storage of the state at the base of the stage, allocated during regularization
#####

# Each scheme is rebuilt with the buffers it needs, each allocated by `buffer()`
function materialize_radiation_storage(scheme::AbstractRadiationScheme, grid, loc, dim)
    FT = eltype(grid)
    Sx, Sy, Sz = size(grid, loc) # loc-aware: Face on Bounded gives N+1, matching the kernel range
    Hx, Hy, _ = halo_size(grid)
    arch = architecture(grid)

    tangential_ranges = dim == 1 ? (connected_range(loc[2], topology(grid, 2), Sy, Hy), 1:Sz) :
                        dim == 2 ? (connected_range(loc[1], topology(grid, 1), Sx, Hx), 1:Sz) :
                                   (1:Sx, 1:Sy)

    buffer() = StaticStridesOffsetArray(zeros(arch, FT, map(length, tangential_ranges)...), map(r -> first(r) - 1, tangential_ranges)...)
    return radiation_storage(scheme, buffer)
end

rebuild_classification(::Value, scheme)      = Value(scheme)
rebuild_classification(::NormalFlow, scheme) = NormalFlow(scheme)

function regularize_boundary_condition(bc::RBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    scheme = materialize_radiation_storage(bc.classification.scheme, grid, loc, dim)
    return BoundaryCondition(rebuild_classification(bc.classification, scheme), regularized_condition)
end

#####
##### The halo fill leaves the exterior state in place
#####

@inline   _fill_west_halo!(j, k, grid, c, ::RBC, loc, args...) = nothing
@inline   _fill_east_halo!(j, k, grid, c, ::RBC, loc, args...) = nothing
@inline  _fill_south_halo!(i, k, grid, c, ::RBC, loc, args...) = nothing
@inline  _fill_north_halo!(i, k, grid, c, ::RBC, loc, args...) = nothing
@inline _fill_bottom_halo!(i, j, grid, c, ::RBC, loc, args...) = nothing
@inline    _fill_top_halo!(i, j, grid, c, ::RBC, loc, args...) = nothing

#####
##### Stepping, caching and initializing the exterior state, on the boundary plane of the halo fill
#####

# `i, j` index the boundary plane, as in `getbc`
@kernel function _update_exterior_state!(update!, c, side, bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    update!(side, i, j, grid, c, bc, loc, args...)
end

function update_exterior_state!(update!, bc, side, field, args...)
    grid = field.grid
    loc = instantiated_location(field)
    size = fill_halo_size(field.data, side, field.indices, bc, loc, grid)
    offset = fill_halo_offset(size, side, field.indices)
    size, offset = connected_fill_extent(side, bc, size, offset, field.indices, loc, grid)
    launch!(architecture(grid), grid, kernel_parameters(size, offset), _update_exterior_state!, update!, field.data, side, bc, loc, grid, args)
    return nothing
end

step_boundary_condition!(bc::RBC, side, field, clock, model_fields, Δt) = update_exterior_state!(step_exterior_state!, bc, side, field, clock, model_fields, Δt)
cache_boundary_condition!(bc::RBC, side, field, clock, model_fields) = update_exterior_state!(cache_exterior_state!, bc, side, field)
initialize_boundary_condition!(bc::RBC, side, field, clock, model_fields) = update_exterior_state!(initialize_exterior_state!, bc, side, field, clock, model_fields)

@inline function cache_exterior_state!(side, i, j, grid, c, bc, loc)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    @inbounds bc.classification.scheme.φ⁻[i, j] = c[iᵇ, jᵇ, kᵇ]
    return nothing
end

# The initial condition sets a boundary-normal velocity on the boundary face; other fields start with a zero gradient
@inline initialize_exterior_state!(side, i, j, grid, c, ::RNFBC, loc, clock, model_fields) = nothing

@inline function initialize_exterior_state!(side, i, j, grid, c, bc, loc, clock, model_fields)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)
    @inbounds c[iᵇ, jᵇ, kᵇ] = c[i₁, j₁, k₁]
    return nothing
end

# A `target_transport` is `nothing`, a fixed transport, or a callable of the grid (kept as is).
convert_target_transport(FT, ::Nothing) = nothing
convert_target_transport(FT, target_transport::Number) = convert(FT, target_transport)
convert_target_transport(FT, target_transport) = target_transport

using Adapt: Adapt
using Oceananigans.Fields: Field

import Oceananigans: prognostic_state, restore_prognostic_state!

"""
    struct ForwardBackwardScheme

A timestepping scheme used for substepping in the split-explicit free surface solver.

The equations are evolved as follows:
```math
\\begin{gather}
U^{m+1} = U^m - Δτ (∂_x η^m - G^U), \\\\
V^{m+1} = V^m - Δτ (∂_y η^m - G^V), \\\\
η^{m+1} = η^m - Δτ (∂_x U^{m+1} + ∂_y V^{m+1}).
\\end{gather}
```
"""
struct ForwardBackwardScheme end

materialize_timestepper(::ForwardBackwardScheme, grid, args...) = ForwardBackwardScheme()

"""
    struct DissipativeForwardBackwardScheme

A dissipative forward-backward timestepping scheme for barotropic substepping in the
split-explicit free surface solver. Temporal extrapolation adds controlled dissipation
to the barotropic mode, eliminating the need for time-averaging of the barotropic
velocities and free surface.

The equations are evolved as follows:
```math
\\begin{gather}
η^\\star = (1 + θ) η^m - θ \\, η^{m-1}, \\\\
U^{m+1} = U^m + Δτ \\left(- g H ∂_x η^\\star + G^U \\right), \\\\
V^{m+1} = V^m + Δτ \\left(- g H ∂_y η^\\star + G^V \\right), \\\\
U^\\star = (1 + α) U^{m+1} - α \\, U^m, \\\\
V^\\star = (1 + α) V^{m+1} - α \\, V^m, \\\\
η^{m+1} = η^m - Δτ \\left(∂_x U^\\star + ∂_y V^\\star \\right).
\\end{gather}
```

The parameters ``θ`` and ``α`` control the dissipation applied to the free surface and
barotropic velocities respectively. With this scheme the barotropic velocities and free
surface at the end of subcycling are used directly (no time-averaging is needed).
Transport velocities are still accumulated using transport weights.
"""
struct DissipativeForwardBackwardScheme{FT, E, U, V}
    θ :: FT   # Dissipation parameter for η extrapolation
    α :: FT   # Dissipation parameter for U, V extrapolation
    ηᵐ⁻¹ :: E # Previous substep free surface
    Uᵐ :: U   # Pre-update zonal barotropic velocity
    Vᵐ :: V   # Pre-update meridional barotropic velocity
end

"""
    DissipativeForwardBackwardScheme(; θ = 0.1, α = 0.1)

Construct a `DissipativeForwardBackwardScheme` with dissipation parameters `θ` and `α`.
"""
DissipativeForwardBackwardScheme(; θ = 0.14, α = 0.14) =
    DissipativeForwardBackwardScheme(θ, α, nothing, nothing, nothing)

function materialize_timestepper(scheme::DissipativeForwardBackwardScheme, grid, free_surface, velocities, u_bcs, v_bcs)
    FT = eltype(grid)
    θ = convert(FT, scheme.θ)
    α = convert(FT, scheme.α)
    ηᵐ⁻¹ = free_surface_displacement_field(velocities, free_surface, grid)
    Uᵐ = Field{Face, Center, Nothing}(grid)
    Vᵐ = Field{Center, Face, Nothing}(grid)
    return DissipativeForwardBackwardScheme(θ, α, ηᵐ⁻¹, Uᵐ, Vᵐ)
end

Adapt.adapt_structure(to, ts::DissipativeForwardBackwardScheme) =
    DissipativeForwardBackwardScheme(ts.θ, ts.α,
                                     Adapt.adapt(to, ts.ηᵐ⁻¹),
                                     Adapt.adapt(to, ts.Uᵐ),
                                     Adapt.adapt(to, ts.Vᵐ))

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardScheme, args...) = nothing

# The functions `η★` `U★` represent the value of free surface and barotropic velocity at time step m+1/2
@inline η★(i, j, k, grid,  ::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]
@inline U★(i, j, k, grid,  ::ForwardBackwardScheme, Uᵐ)   = @inbounds Uᵐ[i, j, k]
@inline V★(i, j, k, grid,  ::ForwardBackwardScheme, Vᵐ)   = @inbounds Vᵐ[i, j, k]

@inline cache_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η)    = nothing
@inline   cache_previous_velocities!(::ForwardBackwardScheme, i, j, k, U, V) = nothing

##### DissipativeForwardBackwardScheme extrapolations

# η★ = (1+θ)ηᵐ - θ·ηᵐ⁻¹
@inline η★(i, j, k, grid, ts::DissipativeForwardBackwardScheme, η) = @inbounds (1 + ts.θ) * η[i, j, k] - ts.θ * ts.ηᵐ⁻¹[i, j, k]
@inline U★(i, j, k, grid, ts::DissipativeForwardBackwardScheme, U) = @inbounds (1 + ts.α) * U[i, j, k] - ts.α * ts.Uᵐ[i, j, k]
@inline V★(i, j, k, grid, ts::DissipativeForwardBackwardScheme, V) = @inbounds (1 + ts.α) * V[i, j, k] - ts.α * ts.Vᵐ[i, j, k]

# Cache ηᵐ before updating η (becomes ηᵐ⁻¹ for the next substep's η★)
@inline function cache_previous_free_surface!(ts::DissipativeForwardBackwardScheme, i, j, k, η)
    @inbounds ts.ηᵐ⁻¹[i, j, k] = η[i, j, k]
end

# Cache Uᵐ, Vᵐ before updating U, V (used in U★ computation)
@inline function cache_previous_velocities!(ts::DissipativeForwardBackwardScheme, i, j, k, U, V)
    @inbounds ts.Uᵐ[i, j, k] = U[i, j, k]
    @inbounds ts.Vᵐ[i, j, k] = V[i, j, k]
end

# Initialize cached state = current state so that the first substep has no extrapolation
function initialize_free_surface_timestepper!(ts::DissipativeForwardBackwardScheme, η, U, V)
    parent(ts.ηᵐ⁻¹) .= parent(η)
    parent(ts.Uᵐ) .= parent(U)
    parent(ts.Vᵐ) .= parent(V)
    return nothing
end

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(restored::ForwardBackwardScheme, ::Nothing) = restored

prognostic_state(ts::DissipativeForwardBackwardScheme) =
    (ηᵐ⁻¹ = prognostic_state(ts.ηᵐ⁻¹),
     Uᵐ   = prognostic_state(ts.Uᵐ),
     Vᵐ   = prognostic_state(ts.Vᵐ))

function restore_prognostic_state!(restored::DissipativeForwardBackwardScheme, from)
    restore_prognostic_state!(restored.ηᵐ⁻¹, from.ηᵐ⁻¹)
    restore_prognostic_state!(restored.Uᵐ, from.Uᵐ)
    restore_prognostic_state!(restored.Vᵐ, from.Vᵐ)
    return restored
end

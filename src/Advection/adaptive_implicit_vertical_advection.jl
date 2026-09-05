using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶠᶜᶠ, Δzᶜᶠᶠ, Az_qᶜᶜᶠ, Azᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: Center, Face
using Oceananigans.BoundaryConditions: BoundaryConditions, _unwrap_for_gpu
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper, RungeKutta3TimeStepper

const AVID = AdaptiveVerticallyImplicitDiscretization

#####
##### Explicit velocity scaling
#####
##### The explicit vertical velocity is wᵉ = w / f(α, cfl) where
##### f = max(1, α / cfl) and α = |w| * Δt / Δz.
##### This ensures the explicit CFL is always ≤ cfl.
#####

# Scale factor: min(1, cfl * Δz / (|w| * Δt))
# When |w| * Δt / Δz ≤ cfl: scale = 1 (fully explicit)
# When |w| * Δt / Δz > cfl: scale = cfl * Δz / (|w| * Δt) < 1
@inline function explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    w  = @inbounds W[i, j, k]
    α  = abs(w) * Δt / Δz
    return ifelse(α > td.cfl, td.cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return ifelse(α > td.cfl, td.cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return ifelse(α > td.cfl, td.cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶜᶜᶜ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    w  = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return ifelse(α > td.cfl, td.cfl / α, one(α))
end

#####
##### Flux dispatch, use scaled w velocities for explicit fluxes
#####

# Horizontal advection is fully explicit with AVID
@inline advective_tracer_flux_x(i, j, k, grid, scheme, ::AVID, U, c) = advective_tracer_flux_x(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme, ::AVID, V, c) = advective_tracer_flux_y(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, c)

@inline function advective_tracer_flux_z(i, j, k, grid, scheme, td::AVID, W, c)
    s = explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_tracer_flux_z(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, c)
end

# Horizontal momentum fluxes are fully explicit with AVID
@inline advective_momentum_flux_Uu(i, j, k, grid, scheme, ::AVID, U, u) = advective_momentum_flux_Uu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme, ::AVID, V, u) = advective_momentum_flux_Vu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, u)
@inline advective_momentum_flux_Uv(i, j, k, grid, scheme, ::AVID, U, v) = advective_momentum_flux_Uv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme, ::AVID, V, v) = advective_momentum_flux_Vv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, v)
@inline advective_momentum_flux_Uw(i, j, k, grid, scheme, ::AVID, U, w) = advective_momentum_flux_Uw(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme, ::AVID, V, w) = advective_momentum_flux_Vw(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, w)

# Vertical advection of momentum: scale by explicit_velocity_scale.
@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme, td::AVID, W, u)
    s  = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, u)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme, td::AVID, W, v)
    s  = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, v)
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme, td::AVID, W, w)
    s  = explicit_velocity_scaleᶜᶜᶜ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Ww(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, w)
end

#####
##### Utility functions
#####

BoundaryConditions.needs_implicit_solver(::AdaptiveImplicitVerticalAdvection) = true

# `any` follows the three-valued logic and _may_ return `missing` in some cases.  Let's
# inform the compiler with the `::Bool` annotation that we know we only deal with booleans.
BoundaryConditions.needs_implicit_solver(a::NamedTuple) = any(BoundaryConditions.needs_implicit_solver, values(a))::Bool

"""
$(TYPEDSIGNATURES)

Refresh the state an advection scheme carries between stages, before tendencies are computed. `tracer` is the
field the scheme advects, or `nothing` for the `momentum` entry.
"""
update_advection!(advection, model) = nothing

# `advection` is `(momentum, tracer_names...)`, with the tracer entries in `model.tracers` order.
@inline function update_advection!(advection::NamedTuple, model)
    update_advection!(advection.momentum, model, nothing)
    return update_tracer_advection!(Base.tail(values(advection)), values(model.tracers), model)
end

@inline update_tracer_advection!(::Tuple{}, ::Tuple{}, model) = nothing

@inline function update_tracer_advection!(schemes::Tuple, tracers::Tuple, model)
    update_advection!(first(schemes), model, first(tracers))
    return update_tracer_advection!(Base.tail(schemes), Base.tail(tracers), model)
end

@inline function update_advection!(scheme, model, tracer)
    update_adaptive_timestep!(scheme, model)
    update_bounds_preserving_limiter!(scheme, model.grid, tracer)
    return nothing
end

update_advection!(scheme::FluxFormAdvection, model, tracer) = update_advection!(scheme.z, model, tracer)

update_adaptive_timestep!(scheme, model) = nothing

@inline function update_adaptive_timestep!(scheme::AdaptiveImplicitVerticalAdvection, model)
    td = TimeSteppers.time_discretization(scheme)
    td.Δt[] = adaptive_advection_timestep(model.timestepper, model.clock)
    return nothing
end

# Δτ of the *next* substep, so that wᵉ in Gⁿ matches the next wⁱ.
@inline adaptive_advection_timestep(timestepper, clock) = clock.last_Δt

@inline function adaptive_advection_timestep(timestepper::SplitRungeKuttaTimeStepper, clock)
    stage  = clock.stage
    Δt     = clock.last_stage_Δt * timestepper.β[stage]
    nstage = ifelse(stage < timestepper.Nstages, stage + 1, 1)
    return Δt / timestepper.β[nstage]
end

@inline function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper::SSPRungeKuttaTimeStepper, clock)
    td = TimeSteppers.time_discretization(a)
    td.Δt[] = clock.last_stage_Δt
    return nothing
end

@inline sum_rk3_coefficients(ts, ::Val{1}) = ts.γ¹
@inline sum_rk3_coefficients(ts, ::Val{2}) = ts.γ² + ts.ζ²
@inline sum_rk3_coefficients(ts, ::Val{3}) = ts.γ¹ + ts.ζ³

@inline function adaptive_advection_timestep(timestepper::RungeKutta3TimeStepper, clock)
    stage  = clock.stage
    nstage = stage == 3 ? 1 : stage + 1
    Δt     = clock.last_stage_Δt / sum_rk3_coefficients(timestepper, Val(stage))
    return Δt * sum_rk3_coefficients(timestepper, Val(nstage))
end

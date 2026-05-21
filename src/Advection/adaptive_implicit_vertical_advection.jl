using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶠᶜᶠ, Δzᶜᶠᶠ, Az_qᶜᶜᶠ, Azᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: Center, Face
using Oceananigans.BoundaryConditions: _unwrap_for_gpu
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, RungeKutta3TimeStepper

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

# Horizontal momentum fluxes (and Ww) are fully explicit with AVID
@inline advective_momentum_flux_Uu(i, j, k, grid, scheme, ::AVID, U, u) = advective_momentum_flux_Uu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, scheme, ::AVID, V, u) = advective_momentum_flux_Vu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, u)
@inline advective_momentum_flux_Uv(i, j, k, grid, scheme, ::AVID, U, v) = advective_momentum_flux_Uv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, scheme, ::AVID, V, v) = advective_momentum_flux_Vv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, v)
@inline advective_momentum_flux_Uw(i, j, k, grid, scheme, ::AVID, U, w) = advective_momentum_flux_Uw(i, j, k, grid, scheme, ExplicitTimeDiscretization(), U, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, scheme, ::AVID, V, w) = advective_momentum_flux_Vw(i, j, k, grid, scheme, ExplicitTimeDiscretization(), V, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, scheme, ::AVID, W, w) = advective_momentum_flux_Ww(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, w)

# Vertical advection of horizontal momentum: scale by explicit_velocity_scale.
@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme, td::AVID, W, u)
    s  = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, u)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme, td::AVID, W, v)
    s  = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, v)
end

#####
##### Utility functions
#####

needs_implicit_solver(advection) = false
needs_implicit_solver(::AdaptiveImplicitVerticalAdvection) = true
needs_implicit_solver(a::NamedTuple) = any(needs_implicit_solver, values(a))

"""
    update_advection_timestep!(advection, timestepper, clock)

Set `advection.Δt[]` to the next substep's Δτ so wᵉ in Gⁿ matches the next wⁱ.
"""
update_advection_timestep!(advection, timestepper, clock) = nothing

function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper, clock)
    td = TimeSteppers.time_discretization(a)
    td.Δt[] = clock.last_Δt
    return nothing
end

@inline function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper::SplitRungeKuttaTimeStepper, clock)
    td      = TimeSteppers.time_discretization(a)
    stage   = clock.stage
    Δt      = clock.last_stage_Δt * timestepper.β[stage]
    nstage  = ifelse(stage < timestepper.Nstages, stage + 1, 1)
    td.Δt[] = Δt / timestepper.β[nstage]
    return nothing
end

@inline sum_rk3_coefficients(ts, ::Val{1}) = ts.γ¹
@inline sum_rk3_coefficients(ts, ::Val{2}) = ts.γ² + ts.ζ²
@inline sum_rk3_coefficients(ts, ::Val{3}) = ts.γ¹ + ts.ζ³

@inline function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper::RungeKutta3TimeStepper, clock)
    td      = TimeSteppers.time_discretization(a)
    stage   = clock.stage
    nstage  = stage == 3 ? 1 : stage + 1
    Δt      = clock.last_stage_Δt / sum_rk3_coefficients(timestepper, Val(stage))
    td.Δt[] = Δt * sum_rk3_coefficients(timestepper, Val(nstage))
    return nothing
end

update_advection_timestep!(a::FluxFormAdvection, timestepper, clock) = update_advection_timestep!(a.z, timestepper, clock)

function update_advection_timestep!(a::NamedTuple, timestepper, clock)
    for scheme in values(a)
        update_advection_timestep!(scheme, timestepper, clock)
    end
    return nothing
end

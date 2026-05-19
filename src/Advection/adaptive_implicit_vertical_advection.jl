using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶠᶜᶠ, Δzᶜᶠᶠ, Az_qᶜᶜᶠ, Azᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: Center, Face
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper

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
@inline function explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, vd, W)
    Δt = vd.Δt[]
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    w  = @inbounds W[i, j, k]
    α  = abs(w) * Δt / Δz
    return ifelse(α > vd.cfl, vd.cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, vd, W)
    Δt = vd.Δt[]
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return ifelse(α > vd.cfl, vd.cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, vd, W)
    Δt = vd.Δt[]
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    w  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    α  = abs(w) * Δt / Δz
    return ifelse(α > vd.cfl, vd.cfl / α, one(α))
end

#####
##### Flux dispatch
#####
##### Horizontal fluxes pass through to the explicit_scheme unchanged.
##### Vertical tracer flux uses the CFL-scaled velocity wᵉ.
##### Vertical momentum fluxes also pass through to the explicit scheme
##### (implicit treatment is only for tracers and horizontal velocities).
#####

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::AVID, W, c)
    s = explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, vd, W)
    return s * advective_tracer_flux_z(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, c)
end

# Vertical advection of horizontal momentum: scale by explicit_velocity_scale.
@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme, vd::AVID, W, u)
    s  = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, vd, W)
    return s * advective_momentum_flux_Wu(i, j, k, grid, ExplicitTimeDiscretization(), scheme, W, u)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme, vd::AVID, W, v)
    s  = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, vd, W)
    return s * advective_momentum_flux_Wv(i, j, k, grid, ExplicitTimeDiscretization(), scheme, W, v)
end

#####
##### Utility functions
#####

needs_implicit_solver(advection) = false
needs_implicit_solver(::AdaptiveImplicitVerticalAdvection) = true
needs_implicit_solver(a::FluxFormAdvection) = needs_implicit_solver(a.z)
needs_implicit_solver(a::NamedTuple) = any(needs_implicit_solver, values(a))

"""
    update_advection_timestep!(advection, timestepper, stage, Δτ)

Set `advection.Δt[]` to the next substep's Δτ so wᵉ in Gⁿ matches the next wⁱ.
"""
update_advection_timestep!(advection, timestepper, stage, Δt) = nothing

function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper, stage, Δt)
    a.vd.Δt[] = Δt
    return nothing
end

@inline function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper::SplitRungeKuttaTimeStepper, stage, Δτ)
    Δt     = Δτ * timestepper.β[stage]
    nstage = ifelse(stage < timestepper.Nstages, stage + 1, 1)
    a.vd.Δt[] = Δt / timestepper.β[nstage]
    return nothing
end

update_advection_timestep!(a::FluxFormAdvection, timestepper, stage, Δt) = update_advection_timestep!(a.z, timestepper, stage, Δt)

function update_advection_timestep!(a::NamedTuple, timestepper, stage, Δt)
    for scheme in values(a)
        update_advection_timestep!(scheme, timestepper, stage, Δt)
    end
    return nothing
end

# `nothing` Δτ disambiguation
update_advection_timestep!(advection, timestepper, stage, ::Nothing) = nothing
update_advection_timestep!(::AdaptiveImplicitVerticalAdvection, timestepper, stage, ::Nothing) = nothing
update_advection_timestep!(::AdaptiveImplicitVerticalAdvection, ::SplitRungeKuttaTimeStepper, stage, ::Nothing) = nothing
update_advection_timestep!(::FluxFormAdvection, timestepper, stage, ::Nothing) = nothing
update_advection_timestep!(::NamedTuple, timestepper, stage, ::Nothing) = nothing

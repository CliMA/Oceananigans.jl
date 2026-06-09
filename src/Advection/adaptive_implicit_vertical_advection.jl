using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶠᶜᶠ, Δzᶜᶠᶠ, Az_qᶜᶜᶠ, Azᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Grids: Center, Face, inactive_cell
using Oceananigans.BoundaryConditions: _unwrap_for_gpu
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, RungeKutta3TimeStepper
using GPUArraysCore: @allowscalar
using Statistics: median, quantile

const AVID = AdaptiveVerticallyImplicitDiscretization

@inline resolved_cfl(td) = _unwrap_for_gpu(td.cfl)

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
    w = @inbounds W[i, j, k]
    α = abs(w) * Δt / Δz
    cfl = resolved_cfl(td)
    return ifelse(α > cfl, cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    w = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    α = abs(w) * Δt / Δz
    cfl = resolved_cfl(td)
    return ifelse(α > cfl, cfl / α, one(α))
end

@inline function explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    Δt = _unwrap_for_gpu(td.Δt)
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    w = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    α = abs(w) * Δt / Δz
    cfl = resolved_cfl(td)
    return ifelse(α > cfl, cfl / α, one(α))
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
    s = explicit_velocity_scaleᶠᶜᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wu(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, u)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme, td::AVID, W, v)
    s = explicit_velocity_scaleᶜᶠᶠ(i, j, k, grid, scheme, td, W)
    return s * advective_momentum_flux_Wv(i, j, k, grid, scheme, ExplicitTimeDiscretization(), W, v)
end

#####
##### Utility functions
#####

needs_implicit_solver(advection) = false
needs_implicit_solver(::AdaptiveImplicitVerticalAdvection) = true
needs_implicit_solver(a::NamedTuple) = any(needs_implicit_solver, values(a))

@inline function local_vertical_cflᶜᶜᶜ(i, j, k, grid, W, Δt)
    inactive_cell(i, j, k, grid) && return NaN

    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    w = @allowscalar W[i, j, k]
    α = abs(w) * Δt / Δz
    return ifelse(isfinite(α), α, NaN)
end

@inline function push_local_vertical_cfl!(cfl_values, i, j, k, grid, W, Δt)
    α = local_vertical_cflᶜᶜᶜ(i, j, k, grid, W, Δt)
    isfinite(α) && push!(cfl_values, α)
    return nothing
end

function sampled_vertical_cfl_values(model, Δt, top_levels, bottom_levels)
    grid = model.grid
    W = model.velocities.w
    Nx, Ny, Nz = size(grid)
    cfl_values = Float64[]

    @inbounds for j in 1:Ny, i in 1:Nx
        wet_levels = Int[]

        for k in 1:Nz
            inactive_cell(i, j, k, grid) || push!(wet_levels, k)
        end

        isempty(wet_levels) && continue

        sampled_levels = Int[]

        ntop = min(top_levels, length(wet_levels))
        for offset in 0:ntop-1
            push!(sampled_levels, wet_levels[end - offset])
        end

        nbottom = min(bottom_levels, length(wet_levels))
        for offset in 0:nbottom-1
            k = wet_levels[begin + offset]
            k in sampled_levels || push!(sampled_levels, k)
        end

        for k in sampled_levels
            push_local_vertical_cfl!(cfl_values, i, j, k, grid, W, Δt)
        end
    end

    return cfl_values
end

function update_adaptive_implicit_vertical_advection_diagnostics!(td::AVID, model)
    Δt = td.Δt[]
    cfl_values = sampled_vertical_cfl_values(model, Δt, td.sample_top_levels, td.sample_bottom_levels)
    FT = typeof(td.Δt[])

    if isempty(cfl_values)
        td.cfl[] = isnothing(td.maximum_explicit_cfl) ? zero(FT) : td.maximum_explicit_cfl
        td.realized_implicit_fraction[] = zero(FT)
        td.median_cfl[] = zero(FT)
        td.max_cfl[] = zero(FT)
        return nothing
    end

    td.median_cfl[] = convert(FT, median(cfl_values))
    td.max_cfl[] = convert(FT, maximum(cfl_values))

    threshold = if !isnothing(td.maximum_explicit_cfl)
        td.maximum_explicit_cfl
    else
        quantile_level = one(FT) - td.implicit_fraction
        convert(FT, quantile(cfl_values, quantile_level))
    end

    td.cfl[] = threshold
    td.realized_implicit_fraction[] = convert(FT, count(>(threshold), cfl_values) / length(cfl_values))

    return nothing
end

@inline function set_advection_timestep!(td, timestepper, clock)
    td.Δt[] = clock.last_Δt
    return nothing
end

@inline function set_advection_timestep!(td, timestepper::SplitRungeKuttaTimeStepper, clock)
    stage = clock.stage
    Δt = clock.last_stage_Δt * timestepper.β[stage]
    nstage = ifelse(stage < timestepper.Nstages, stage + 1, 1)
    td.Δt[] = Δt / timestepper.β[nstage]
    return nothing
end

@inline sum_rk3_coefficients(ts, ::Val{1}) = ts.γ¹
@inline sum_rk3_coefficients(ts, ::Val{2}) = ts.γ² + ts.ζ²
@inline sum_rk3_coefficients(ts, ::Val{3}) = ts.γ¹ + ts.ζ³

@inline function set_advection_timestep!(td, timestepper::RungeKutta3TimeStepper, clock)
    stage = clock.stage
    nstage = stage == 3 ? 1 : stage + 1
    Δt = clock.last_stage_Δt / sum_rk3_coefficients(timestepper, Val(stage))
    td.Δt[] = Δt * sum_rk3_coefficients(timestepper, Val(nstage))
    return nothing
end

"""
    update_advection_timestep!(advection, timestepper, clock)

Set `advection.Δt[]` to the next substep's Δτ so wᵉ in Gⁿ matches the next wⁱ.
"""
update_advection_timestep!(advection, timestepper, clock) = nothing
update_advection_timestep!(advection, timestepper, clock, model) = update_advection_timestep!(advection, timestepper, clock)

function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper, clock)
    td = TimeSteppers.time_discretization(a)
    set_advection_timestep!(td, timestepper, clock)
    return nothing
end

function update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, timestepper, clock, model)
    td = TimeSteppers.time_discretization(a)
    set_advection_timestep!(td, timestepper, clock)
    update_adaptive_implicit_vertical_advection_diagnostics!(td, model)
    return nothing
end

update_advection_timestep!(a::FluxFormAdvection, timestepper, clock) = update_advection_timestep!(a.z, timestepper, clock)
update_advection_timestep!(a::FluxFormAdvection, timestepper, clock, model) = update_advection_timestep!(a.z, timestepper, clock, model)

function update_advection_timestep!(a::NamedTuple, timestepper, clock)
    for scheme in values(a)
        update_advection_timestep!(scheme, timestepper, clock)
    end
    return nothing
end

function update_advection_timestep!(a::NamedTuple, timestepper, clock, model)
    for scheme in values(a)
        update_advection_timestep!(scheme, timestepper, clock, model)
    end
    return nothing
end

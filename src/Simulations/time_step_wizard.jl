mutable struct TimeStepWizard{FT, C, D}
                         cfl :: FT
               diffusive_cfl :: FT
                  max_change :: FT
                  min_change :: FT
                      max_Δt :: FT
                      min_Δt :: FT
    cell_advection_timescale :: C
    cell_diffusion_timescale :: D
end

infinite_diffusion_timescale(args...) = Inf # its not very limiting

"""
    TimeStepWizard(cfl=0.2, diffusive_cfl=Inf, max_change=1.1, min_change=0.5, max_Δt=Inf, min_Δt=0.0)

Callback for adapting simulation time-steps `Δt` to maintain the advective
Courant-Freidrichs-Lewy (`cfl`) number, the `diffusive_cfl`, while maintaining
`max_Δt`, `min_Δt`, and satisfying `max_change` and `min_change` criteria
so `Δt` is not adapted "too quickly".

For more information on `cfl`, see
https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition.

Example
=======

To use `TimeStepWizard`, adapt in a [`Callback`](@ref) and add it to a `Simulation`:

```julia
julia> simulation = Simulation(model, Δt=0.9, stop_iteration=100)

julia> wizard = TimeStepWizard(cfl=0.2)

julia> simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
```

Then when `run!(simulation)` is invoked, the time-step `simulation.Δt` will be updated every 4 iterations.
Note that the name `:wizard` is unimportant.
"""
function TimeStepWizard(FT=Float64; cfl = 0.2,
                                    diffusive_cfl = Inf,
                                    max_change = 1.1,
                                    min_change = 0.5,
                                    max_Δt = Inf,
                                    min_Δt = 0.0,
                                    cell_advection_timescale = cell_advection_timescale,
                                    cell_diffusion_timescale = infinite_diffusion_timescale)

    isfinite(diffusive_cfl) && # user wants to limit by diffusive CFL
    !(cell_diffusion_timescale === infinite_diffusion_timescale) && # user did not provide custom timescale
        (cell_diffusion_timescale = Oceananigans.TurbulenceClosures.cell_diffusion_timescale)

    C = typeof(cell_advection_timescale)
    D = typeof(cell_diffusion_timescale)

    return TimeStepWizard{FT, C, D}(cfl, diffusive_cfl, max_change, min_change, max_Δt, min_Δt,
                                    cell_advection_timescale, cell_diffusion_timescale)
end

using Oceananigans.Grids: topology

"""
     new_time_step(old_Δt, wizard, model)

Return a new time_step given `model.velocities` and model diffusivites,
and the parameters of the `TimeStepWizard` `wizard`.
"""
function new_time_step(old_Δt, wizard, model)

    advective_Δt = wizard.cfl * wizard.cell_advection_timescale(model)
    diffusive_Δt = wizard.diffusive_cfl * wizard.cell_diffusion_timescale(model)

    new_Δt = min(advective_Δt, diffusive_Δt)

    # Put the kibosh on if needed
    new_Δt = min(wizard.max_change * old_Δt, new_Δt)
    new_Δt = max(wizard.min_change * old_Δt, new_Δt)
    new_Δt = clamp(new_Δt, wizard.min_Δt, wizard.max_Δt)

    return new_Δt
end

(wizard::TimeStepWizard)(simulation) =
    simulation.Δt = new_time_step(simulation.Δt, wizard, simulation.model)


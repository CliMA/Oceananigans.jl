using Oceananigans: TurbulenceClosures
using Oceananigans.Grids: prettysummary

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

Base.summary(wizard::TimeStepWizard) = string("TimeStepWizard(",
                                                "cfl=",           prettysummary(wizard.cfl),
                                              ", max_Δt=",        prettysummary(wizard.max_Δt),
                                              ", min_Δt=",        prettysummary(wizard.min_Δt), ")")

"""
    TimeStepWizard([FT=Float64;]
                   cfl = 0.2,
                   diffusive_cfl = Inf,
                   max_change = 1.1,
                   min_change = 0.5,
                   max_Δt = Inf,
                   min_Δt = 0.0,
                   cell_advection_timescale = cell_advection_timescale,
                   cell_diffusion_timescale = infinite_diffusion_timescale)

Callback for adapting simulation to maintain the advective Courant-Freidrichs-Lewy (CFL)
number to `cfl`, the `diffusive_cfl`, while also maintaining `max_Δt`, `min_Δt`, and
satisfying `max_change` and `min_change` criteria so that the simulation's timestep
`simulation.Δt` is not adapted "too quickly".

For more information on the CFL number, see its [wikipedia entry]
(https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition).

Example
=======

To use `TimeStepWizard`, adapt in a [`Callback`](@ref) and add it to a `Simulation`:

```julia
julia> simulation = Simulation(model, Δt=0.9, stop_iteration=100)

julia> wizard = TimeStepWizard(cfl=0.2)

julia> simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
```

Then when `run!(simulation)` is invoked, the time-step `simulation.Δt` will be updated every
4 iterations.

(Note that the name `:wizard` is unimportant.)
"""
function TimeStepWizard(FT=Float64;
                        cfl = 0.2,
                        diffusive_cfl = Inf,
                        max_change = 1.1,
                        min_change = 0.5,
                        max_Δt = Inf,
                        min_Δt = 0.0,
                        cell_advection_timescale = cell_advection_timescale,
                        cell_diffusion_timescale = infinite_diffusion_timescale)

    # user wants to limit by diffusive CFL and did not provide custom function to calculate timescale
    if isfinite(diffusive_cfl) && (cell_diffusion_timescale === infinite_diffusion_timescale)
       cell_diffusion_timescale = TurbulenceClosures.cell_diffusion_timescale
    end

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


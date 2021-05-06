mutable struct TimeStepWizard{FT, C, D}
                         cfl :: FT
               diffusive_cfl :: FT
                  max_change :: FT
                  min_change :: FT
                      max_Δt :: FT
                      min_Δt :: FT
                          Δt :: FT
    cell_advection_timescale :: C
    cell_diffusion_timescale :: D
end

infinite_diffusion_timescale(args...) = Inf # its not very limiting

"""
    TimeStepWizard(cfl=0.1, max_change=2.0, min_change=0.5, max_Δt=Inf, min_Δt=0.0, Δt=0.01)

A type for calculating adaptive time steps based on capping the CFL number at `cfl`.

On calling `update_Δt!(wizard, model)`, the `TimeStepWizard` computes a time-step such that
``cfl = max(u/Δx, v/Δy, w/Δz) Δt``, where ``max(u/Δx, v/Δy, w/Δz)`` is the maximum ratio
between model velocity and along-velocity grid spacing anywhere on the model grid. The new
`Δt` is constrained to change by a multiplicative factor no more than `max_change` or no
less than `min_change` from the previous `Δt`, and to be no greater in absolute magnitude
than `max_Δt` and no less than `min_Δt`.
"""
function TimeStepWizard(FT=Float64; cfl = 0.1,
                                    diffusive_cfl = Inf,
                                    max_change = 2.0,
                                    min_change = 0.5,
                                    max_Δt = Inf,
                                    min_Δt = 0.0,
                                    Δt = 0.01,
                                    cell_advection_timescale = cell_advection_timescale,
                                    cell_diffusion_timescale = infinite_diffusion_timescale)

    isfinite(diffusive_cfl) && # user wants to limit by diffusive CFL
    !(cell_diffusion_timescale === infinite_diffusion_timescale) && # user did not provide custom timescale
        (cell_diffusion_timescale = Oceananigans.TurbulenceClosures.cell_diffusion_timescale)

    C = typeof(cell_advection_timescale)
    D = typeof(cell_diffusion_timescale)

    return TimeStepWizard{FT, C, D}(cfl, diffusive_cfl, max_change, min_change, max_Δt, min_Δt, Δt,
                                    cell_advection_timescale, cell_diffusion_timescale)
end

using Oceananigans.Grids: topology

"""
    update_Δt!(wizard, model)

Compute `wizard.Δt` given the velocities and diffusivities of `model`, and the parameters
of `wizard`.
"""
function update_Δt!(wizard, model)

    Δt = min(
             wizard.cfl * wizard.cell_advection_timescale(model),          # advective
             wizard.diffusive_cfl * wizard.cell_diffusion_timescale(model) # diffusive
            )

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = clamp(Δt, wizard.min_Δt, wizard.max_Δt)

    wizard.Δt = Δt

    return nothing
end

(c::CFL{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

get_cfl(Δt, model) = Δt * Umax(model) / Δmin(model.grid)

function cfl_Δt(model, cfl, max_Δt)
    τ = Δmin(model.grid) / Umax(model)
    return min(max_Δt, cfl*τ)
end

function safe_Δt(model, αu, αν=0.01)
    τu = Δmin(model.grid) / Umax(model)
    τν = Δmin(model.grid)^2 / model.closure.ν

    return min(αν*τν, αu*τu)
end

function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = model.grid.Δx
    Δy = model.grid.Δy
    Δz = model.grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

function cell_diffusion_timescale(ν, κ, grid)
    νmax = maximum(abs, ν)
    κmax = maximum(abs, κ)

    Δx = min(model.grid.Δx)
    Δy = min(model.grid.Δy)
    Δz = min(model.grid.Δz)

    Δ = min(Δx, Δy, Δz) # assuming diffusion is isotropic for now

    return min(Δ^2/νmax, Δ^2/κmax)

end

function cell_advection_timescale(model)
    cell_advection_timescale(
                             model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid
                            )
end

function cell_diffusion_timescale(model)
    cell_diffusion_timescale(
                             model.closure.ν,
                             model.closure.κ,
                             model.grid
                            )
end

Base.@kwdef mutable struct TimeStepWizard{T}
              cfl :: T = 0.1
    cfl_diffusion :: T = 2e-2
       max_change :: T = 2.0
       min_change :: T = 0.5
           max_Δt :: T = Inf
               Δt :: T = 0.01
end

function update_Δt!(wizard, model)
    Δt_advection = wizard.cfl           * cell_advection_timescale(model)
    Δt_diffusion = wizard.cfl_diffusion * cell_diffusion_timescale(model)

    # Desired Δt
    Δt = min(Δt_advection, Δt_diffusion)

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = min(wizard.max_Δt, Δt)

    wizard.Δt = Δt

    return nothing
end

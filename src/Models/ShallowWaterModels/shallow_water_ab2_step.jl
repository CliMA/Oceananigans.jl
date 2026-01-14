using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: _ab2_step_field!

import Oceananigans.TimeSteppers: ab2_step!

"""
    ab2_step!(model::ShallowWaterModel, Δt, callbacks)

Perform a single AB2 step for `ShallowWaterModel`.

Advances the solution fields (`uh`, `vh`, `h`) and any tracers using the AB2 scheme:
`U += Δt * ((3/2 + χ) * Gⁿ - (1/2 + χ) * G⁻)`

where `Gⁿ` and `G⁻` are the current and previous tendencies.
"""
function ab2_step!(model::ShallowWaterModel, Δt, callbacks)

    compute_tendencies!(model, callbacks)
    grid = model.grid

    fields = prognostic_fields(model)

    for key in keys(fields)
        launch!(architecture(grid), grid, :xyz, _ab2_step_field!, 
                fields[key],
                Δt,
                model.timestepper.χ,
                model.timestepper.Gⁿ[key],
                model.timestepper.G⁻[key])
    end

    return nothing
end

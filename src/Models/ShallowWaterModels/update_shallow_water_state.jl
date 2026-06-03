using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel, callbacks=[]; compute_tendencies=true)

Update the diagnostic state of `ShallowWaterModel`.

Mask immersed cells for prognostic fields, update model time series,
compute diffusivity fields, fill halo regions for
`model.solution` and `model.tracers`, and compute velocity fields
if using `ConservativeFormulation`.

Next, `callbacks` are executed.

Finally, tendencies are computed if `compute_tendencies=true`.
"""
function update_state!(model::ShallowWaterModel, callbacks=[])

    # Mask immersed fields
    foreach(mask_immersed_field!, merge(model.solution, model.tracers))

    refresh_shallow_water_auxiliary_state!(model)

    foreach(callbacks) do callback
        if isa(callback.callsite, UpdateStateCallsite)
            callback(model)
        end
    end

    return nothing
end

compute_velocities!(U, ::VectorInvariantFormulation) = nothing

function compute_velocities!(U, ::ConservativeFormulation)
    compute!((U.u, U.v))
    return nothing
end

using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model)
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        launch!(model.architecture, model.grid, :xyz, store_field_tendencies!,
                timestepper_previous_tendencies(model.timestepper)[field_name],
                timestepper_tendencies(model.timestepper)[field_name])
    end

    return nothing
end

function store_biogeochemical_tendencies!(model::AbstractModel{<:StrangeSplittingTimeStepper})
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        launch!(model.architecture, model.grid, :xyz, store_field_tendencies!,
                model.timestepper.biogeochemistry.G⁻[field_name],
                model.timestepper.biogeochemistry.Gⁿ[field_name])
    end

    return nothing
end

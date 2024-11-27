using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

has_catke_closure() = false
has_td_closure()    = false

""" Store previous source terms before updating them. """
function store_tendencies!(model)
    model_fields = prognostic_fields(model)

    closure = model.closure

    catke_in_closures = has_catke_closure(closure)
    td_in_closures    = has_td_closure(closure)

    # Tracer update kernels
    for field_name in keys(model_fields)        
        if catke_in_closures && field_name == :e
            @debug "Skipping AB2 step for e"
        elseif td_in_closures && field_name == :ϵ
            @debug "Skipping AB2 step for ϵ"
        elseif td_in_closures && field_name == :e
            @debug "Skipping AB2 step for e"
        else
            launch!(model.architecture, model.grid, :xyz, store_field_tendencies!,
                    model.timestepper.G⁻[field_name],
                    model.timestepper.Gⁿ[field_name])
        end
    end

    return nothing
end

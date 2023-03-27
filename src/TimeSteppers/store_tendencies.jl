using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid

using NVTX
using Oceananigans.Utils: launch!

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model)
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        NVTX.@range "store tendencies for $(field_name)" begin
            launch!(model.architecture, model.grid, :xyz, store_field_tendencies!,
                    model.timestepper.G⁻[field_name],
                    model.grid,
                    model.timestepper.Gⁿ[field_name])
        end
    end

    return nothing
end

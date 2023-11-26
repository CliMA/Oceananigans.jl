using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.ImmersedBoundaries: ActiveCellsIBG
using Oceananigans.Utils: launch!

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, grid::ActiveCellsIBG, G⁰)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_interior_tuple(idx, grid)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model; only_active_cells = only_active_interior_cells(model.grid))
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        launch!(model.architecture, model.grid, :xyz, store_field_tendencies!,
                model.timestepper.G⁻[field_name],
                model.grid,
                model.timestepper.Gⁿ[field_name];
                only_active_cells)
    end

    return nothing
end

using Oceananigans.Grids: active_xspacing, active_yspacing, active_zspacing


import Oceananigans.OutputWriters: save_output!, define_output_variable!
using Oceananigans.OutputWriters: fetch_and_convert_output, drop_output_dims,
                                  netcdf_spatial_dimensions, output_indices
using Oceananigans.Fields: AbstractField
using NCDatasets: defVar

define_timeconstant_variable!(dataset, output::AbstractField, name, array_type, compression, output_attributes, dimensions) =
    defVar(dataset, name, eltype(array_type), netcdf_spatial_dimensions(output),
           compression=compression, attrib=output_attributes)

function save_output!(ds, output, model, ow, name)

    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons...] = data
    return nothing
end

function grid_metric_locations(outputs)
    location_list = []
    for output in values(outputs)
        loc = location(output)
        if (loc ∉ location_list) && (Nothing ∉ loc)
            push!(location_list, location(output))
        end
    end
    return location_list
end

loc_superscript(::Type{Center}) = "ᶜ"
loc_superscript(::Type{Face}) = "ᶠ"
loc_superscript(::Type{Nothing}) = "ⁿ"
function default_grid_spacings(outputs, grid::AbstractRectilinearGrid)
    loc_list = unique(map(location, values(outputs)))
    spacing_operations = Dict()

    for loc in loc_list
        LX, LY, LZ = loc

        # Let's replace Nothing for Center for now since `active_xyzspacing` doesn't accept Nothing as location
        LX = LX == Nothing ? Center : LX
        LY = LY == Nothing ? Center : LY
        LZ = LZ == Nothing ? Center : LZ

        Δx_name = "Δx" * loc_superscript(LX) * "ᵃᵃ"
        Δy_name = "Δyᵃ" * loc_superscript(LY) * "ᵃ"
        Δz_name = "Δzᵃᵃ" * loc_superscript(LZ)

        push!(spacing_operations, 
              Δx_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_xspacing, grid, LX(), LY(), LZ()), dims=(2,3)),
              Δy_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_yspacing, grid, LX(), LY(), LZ()), dims=(1,3)),
              Δz_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_zspacing, grid, LX(), LY(), LZ()), dims=(1,2)))
    end
    return Dict(name => Field(op) for (name, op) in spacing_operations)
end

function write_grid_metrics!(ow, metrics; user_indices = (:, :, :), with_halos=false)
    ds = open(ow)
    @show keys(ds)

    for (metric_name, metric_operation) in metrics
        indices = output_indices(metric_operation, metric_operation.grid, user_indices, with_halos)
        sliced_metric = Field(metric_operation, indices=indices)

        @show metric_name
        define_timeconstant_variable!(ds, sliced_metric, metric_name, ow.array_type, 0, Dict(), ("xC", "yC", "zC"))
        save_output!(ds, sliced_metric, model, ow, metric_name)
    end
    close(ds)
end




using MakieCore

using Oceananigans.Fields: Field, interior_view_indices
using Oceananigans.Grids: nodes, xnodes, ynodes, znodes, halo_size
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range

import MakieCore: convert_arguments, AbstractPlot

function MakieCore.convert_arguments(P::Type{<:AbstractPlot}, f::Field, is = :, js = :, ks = :)
    coordinates = compute_coordinates(is, js, ks, f)

    return convert_arguments(P, coordinates..., interior(f, is, js, ks))
end

compute_coordinates(is, js, ks, f) = (xnodes(f)[is], ynodes(f)[js], znodes(f)[ks])
compute_coordinates(is, js, ks::Number, f) = (xnodes(f)[is], ynodes(f)[js])
compute_coordinates(is, js::Number, ks, f) = (xnodes(f)[is], znodes(f)[ks])
compute_coordinates(is::Number, js, ks, f) = (ynodes(f)[js], znodes(f)[ks])
compute_coordinates(is, js::Number, ks::Number, f) = (xnodes(f)[is], )
compute_coordinates(is::Number, js::Number, ks, f) = (znodes(f)[ks], )
compute_coordinates(is::Number, js::Number, ks::Number, f) = tuple()

function MakieCore.convert_arguments(P::Type{<:AbstractPlot}, fts::FieldTimeSeries, is = :, js = :, ks = :, ns = :)
    coordinates = compute_coordinates(is, js, ks, ns, fts)

    return convert_arguments(P, coordinates..., field_time_series_interior(is, js, ks, ns, fts))
end

compute_coordinates(is, js, ks, f) = (xnodes(f)[is], ynodes(f)[js], znodes(f)[ks])
compute_coordinates(is, js, ks::Number, f) = (xnodes(f)[is], ynodes(f)[js])
compute_coordinates(is, js::Number, ks, f) = (xnodes(f)[is], znodes(f)[ks])
compute_coordinates(is::Number, js, ks, f) = (ynodes(f)[js], znodes(f)[ks])

compute_coordinates(is, js, ks, ns, f) = (compute_coordinates(is, js, ks, f)..., f.times[ns])
compute_coordinates(is, js, ks, f, ::Number) = compute_coordinates(is, js, ks, f)

field_time_series_interior(is, js, ks, ns, fts) = view(interior(fts), is, js, ks, ns)

field_time_series_interior(is, js, ks, ns::Number, fts) = interior(fts[n], is, js, ks)
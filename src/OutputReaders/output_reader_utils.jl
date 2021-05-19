using Oceananigans.Grids: offset_indices

import Oceananigans.Grids: offset_data

function offset_data(underlying_data::AbstractArray{FT, 4}, loc, topo, N, H) where FT
    ii = offset_indices(loc[1], topo[1], N[1], H[1])
    jj = offset_indices(loc[2], topo[2], N[2], H[2])
    kk = offset_indices(loc[3], topo[3], N[3], H[3])
    nn = axes(underlying_data, 4)

    return OffsetArray(underlying_data, ii, jj, kk, nn)
end

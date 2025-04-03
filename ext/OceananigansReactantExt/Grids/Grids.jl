module Grids

export constant_with_arch

using Reactant

using Oceananigans
using Oceananigans: Distributed
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, StaticVerticalDiscretization, MutableVerticalDiscretization
using Oceananigans.Grids: Center, Face, RightConnected, LeftConnected, Periodic, Bounded, Flat
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: GridFittedBottom, AbstractImmersedBoundary

import ..OceananigansReactantExt: deconcretize
import Oceananigans.Grids: LatitudeLongitudeGrid, RectilinearGrid, OrthogonalSphericalShellGrid, size_summary
import Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, TripolarGrid
import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, materialize_immersed_boundary

const ShardedDistributed = Oceananigans.Distributed{<:ReactantState}

const ReactantGrid{FT, TX, TY, TZ} = Union{
    AbstractGrid{FT, TX, TY, TZ, <:ReactantState},
    AbstractGrid{FT, TX, TY, TZ, <:ShardedDistributed}
}

const ReactantImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S} = Union{
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:ReactantState},
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:ShardedDistributed},
}

const ReactantUnderlyingGrid{FT, TX, TY, TZ, CZ} = Union{
    AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:ReactantState},
    AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:ShardedDistributed},
}

const ShardedGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:ShardedDistributed}

function size_summary(grid::ShardedGrid)
    arch = Oceananigans.Architectures.architecture(grid)
    p = arch.partition
    r = Rx, Ry, Rz = ranks(p)
    Nr = prod(r)
    last_rank = Nr - 1

    rank_info = if Nr == 1
        "1 rank"
    else
        "$Nr = $Rx×$Ry×$Rz ranks"
    end

    return string(size_summary(size(grid)), "(", rank_info, ")")
end

include("serial_grids.jl")
include("sharded_grids.jl")

end # module


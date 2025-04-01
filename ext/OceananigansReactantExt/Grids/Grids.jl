module Grids

export constant_with_arch

using Reactant

using Oceananigans
using Oceananigans: Distributed
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, StaticVerticalDiscretization, MutableVerticalDiscretization
using Oceananigans.Grids: Center, Face, RightConnected, LeftConnected, Periodic, Bounded, Flat, BoundedTopology
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: GridFittedBottom, AbstractImmersedBoundary

import ..OceananigansReactantExt: deconcretize
import Oceananigans.Grids: LatitudeLongitudeGrid, RectilinearGrid, OrthogonalSphericalShellGrid
import Oceananigans.Grids: total_length, offset_indices
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

total_length(::Center, ::BoundedTopology, N, H=0) = N + 1 + 2H
offset_indices(::Center, ::BoundedTopology,  N, H=0) = 1 - H : N + H + 1

include("serial_grids.jl")
include("sharded_grids.jl")

end # module


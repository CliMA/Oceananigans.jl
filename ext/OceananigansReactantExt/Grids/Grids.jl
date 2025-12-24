module Grids

export constant_with_arch

using Reactant
using OffsetArrays

using Oceananigans
using Oceananigans: Distributed
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, StaticVerticalDiscretization, MutableVerticalDiscretization
using Oceananigans.Grids: Center, Face, RightConnected, LeftConnected, Periodic, Bounded, Flat, BoundedTopology
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: GridFittedBottom, AbstractImmersedBoundary

import ..OceananigansReactantExt: deconcretize
import Oceananigans.Grids: LatitudeLongitudeGrid, RectilinearGrid, OrthogonalSphericalShellGrid
import Oceananigans.Grids: total_length, offset_data
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

include("serial_grids.jl")
include("sharded_grids.jl")

function total_size(grid::ReactantGrid, loc, indices)
    sz = size(grid)
    halo_sz = halo_size(grid)
    topo = topology(grid)
    return reactant_total_size(loc, topo, sz, halo_sz, indices)
end

function reactant_total_size(loc, topo, sz, halo_sz, indices=default_indices(Val(length(loc))))
    D = length(loc)
    return Tuple(reactant_total_length(instantiate(loc[d]), instantiate(topo[d]), sz[d], halo_sz[d], indices[d]) for d = 1:D)
end

reactant_total_length(loc, topo, N, H, ::Colon) = reactant_total_length(loc, topo, N, H)
reactant_total_length(loc, topo, N, H, ind::AbstractUnitRange) = min(reactant_total_length(loc, topo, N, H), length(ind))
reactant_total_length(loc, topo, N, H) = Oceananigans.Grids.total_length(loc, topo, N, H)
reactant_total_length(::Face, ::BoundedTopology, N, H=0) = N + 2H

reactant_offset_indices(loc, topo, N, H=0) = 1 - H : N + H
reactant_offset_indices(::Nothing, topo, N, H=0) = 1:1
reactant_offset_indices(ℓ,         topo, N, H, ::Colon) = reactant_offset_indices(ℓ, topo, N, H)
reactant_offset_indices(ℓ,         topo, N, H, r::AbstractUnitRange) = r
reactant_offset_indices(::Nothing, topo, N, H, ::AbstractUnitRange) = 1:1

function Oceananigans.Grids.new_data(FT::DataType, arch::Union{ReactantState, ShardedDistributed},
        loc, topo, sz, halo_sz, indices=default_indices(length(loc)))

    Tsz = reactant_total_size(loc, topo, sz, halo_sz, indices)
    underlying_data = zeros(arch, FT, Tsz...)
    indices = validate_indices(indices, loc, topo, sz, halo_sz)

    return offset_data(underlying_data, loc, topo, sz, halo_sz, indices)
end

# The type parameter for indices helps / encourages the compiler to fully type infer `offset_data`
function offset_data(underlying_data::ConcreteRArray, loc, topo, N, H, indices::T=default_indices(length(loc))) where T
    loc = map(instantiate, loc)
    topo = map(instantiate, topo)
    ii = map(reactant_offset_indices, loc, topo, N, H, indices)
    # Add extra indices for arrays of higher dimension than loc, topo, etc.
    # Use the "`ntuple` trick" so the compiler can infer the type of `extra_ii`
    extra_ii = ntuple(Val(ndims(underlying_data)-length(ii))) do i
        Base.@_inline_meta
        axes(underlying_data, i+length(ii))
    end

    return OffsetArray(underlying_data, ii..., extra_ii...)
end

end # module


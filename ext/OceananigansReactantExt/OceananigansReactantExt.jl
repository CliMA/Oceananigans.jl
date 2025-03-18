module OceananigansReactantExt

using Reactant
using Oceananigans
using OffsetArrays

deconcretize(obj) = obj # fallback
deconcretize(a::OffsetArray) = OffsetArray(Array(a.parent), a.offsets...)

include("Utils.jl")
using .Utils

include("Architectures.jl")
using .Architectures

include("Grids.jl")
using .Grids

include("Fields.jl")
using .Fields

include("TimeSteppers.jl")
using .TimeSteppers

include("TurbulenceClosures.jl")
using .TurbulenceClosures

include("Models.jl")
using .Models

include("Simulations/Simulations.jl")
using .Simulations

#####
##### Telling Reactant how to construct types
#####

import ConstructionBase: constructorof 

constructorof(::Type{<:RectilinearGrid{FT, TX, TY, TZ}}) where {FT, TX, TY, TZ} = RectilinearGrid{TX, TY, TZ}
constructorof(::Type{<:VectorInvariant{N, FT, M}}) where {N, FT, M} = VectorInvariant{N, FT, M}

# https://github.com/CliMA/Oceananigans.jl/blob/da9959f3e5d8ee7cf2fb42b74ecc892874ec1687/src/AbstractOperations/conditional_operations.jl#L8
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.ConditionalOperation{LX, LY, LZ, O, F, G, C, M, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, O, F, G, C, M, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)
    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)
    F2 = Reactant.traced_type_inner(F, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    C2 = Reactant.traced_type_inner(C, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(O2)
    return Oceananigans.AbstractOperations.ConditionalOperation{LX2, LY2, LZ2, O2, F2, G2, C2, M2, T2}
end

# https://github.com/CliMA/Oceananigans.jl/blob/da9959f3e5d8ee7cf2fb42b74ecc892874ec1687/src/AbstractOperations/kernel_function_operation.jl#L3
# struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(OA::Type{Oceananigans.AbstractOperations.KernelFunctionOperation{LX, LY, LZ, G, T, K, D}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, G, T, K, D}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    K2 = Reactant.traced_type_inner(K, seen, mode, track_numbers, sharding, runtime)
    D2 = Reactant.traced_type_inner(D, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(G2)
    return Oceananigans.AbstractOperations.KernelFunctionOperation{LX2, LY2, LZ2, G2, T2, K2, D2}
end

# https://github.com/CliMA/Oceananigans.jl/blob/d9b3b142d8252e8e11382d1b3118ac2a092b38a2/src/Grids/orthogonal_spherical_shell_grid.jl#L14
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.Grids.OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch}
    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    Map2 = Reactant.traced_type_inner(Map, seen, mode, track_numbers, sharding, runtime)
    CC2 = Reactant.traced_type_inner(CC, seen, mode, track_numbers, sharding, runtime)
    FC2 = Reactant.traced_type_inner(FC, seen, mode, track_numbers, sharding, runtime)
    CF2 = Reactant.traced_type_inner(CF, seen, mode, track_numbers, sharding, runtime)
    FF2 = Reactant.traced_type_inner(FF2, seen, mode, track_numbers, sharding, runtime)
    FT2 = Base.promote_type(Base.promote_type(Base.promote_type(Base.promote_type(FT2, eltype(CC2)), eltype(FC2)), eltype(CF2)), eltype(FF2))
    return Oceananigans.Grids.OrthogonalSphericalShellGrid{FT2, TX2, TY2, TZ2, Z2, Map2, CC2, FC2, CF2, FF2, Arch}
end

# https://github.com/CliMA/Oceananigans.jl/blob/d9b3b142d8252e8e11382d1b3118ac2a092b38a2/src/ImmersedBoundaries/immersed_boundary_grid.jl#L8
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch}
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    S2 = Reactant.traced_type_inner(S, seen, mode, track_numbers, sharding, runtime)
    FT2 = eltype(G2)
    return Oceananigans.Grids.OrthogonalSphericalShellGrid{FT2, TX2, TY2, TZ2, G2, I2, M2, S2, Arch}
end

# These are additional modules that may need to be Reactantified in the future:
#
# include("Utils.jl")
# include("BoundaryConditions.jl")
# include("Fields.jl")
# include("MultiRegion.jl")
# include("Solvers.jl")
#
# using .Utils
# using .BoundaryConditions
# using .Fields
# using .MultiRegion
# using .Solvers

end # module


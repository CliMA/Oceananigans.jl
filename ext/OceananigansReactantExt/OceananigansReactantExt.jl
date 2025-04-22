module OceananigansReactantExt

using Reactant
using Oceananigans
using OffsetArrays

using Oceananigans: Distributed, DistributedComputations, ReactantState, CPU,
                    OrthogonalSphericalShellGrids
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: Bounded, Periodic, RightConnected

deconcretize(obj) = obj # fallback
deconcretize(a::OffsetArray) = OffsetArray(Array(a.parent), a.offsets...)

include("Utils.jl")
using .Utils

include("Architectures.jl")
using .Architectures

include("Grids/Grids.jl")
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

include("OutputReaders.jl")
using .OutputReaders

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
    @nospecialize(OA::Type{Oceananigans.Grids.OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch, rFT}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch, rFT}
    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    Z2 = Reactant.traced_type_inner(Z, seen, mode, track_numbers, sharding, runtime)
    Map2 = Reactant.traced_type_inner(Map, seen, mode, track_numbers, sharding, runtime)
    CC2 = Reactant.traced_type_inner(CC, seen, mode, track_numbers, sharding, runtime)
    FC2 = Reactant.traced_type_inner(FC, seen, mode, track_numbers, sharding, runtime)
    CF2 = Reactant.traced_type_inner(CF, seen, mode, track_numbers, sharding, runtime)
    FF2 = Reactant.traced_type_inner(FF, seen, mode, track_numbers, sharding, runtime)
    FT2 = Base.promote_type(Base.promote_type(Base.promote_type(Base.promote_type(FT2, eltype(CC2)), eltype(FC2)), eltype(CF2)), eltype(FF2))
    rFT2 = Reactant.traced_type_inner(rFT, seen, mode, track_numbers, sharding, runtime)
    return Oceananigans.Grids.OrthogonalSphericalShellGrid{FT2, TX2, TY2, TZ2, Z2, Map2, CC2, FC2, CF2, FF2, Arch, rFT2}
end

@inline Reactant.make_tracer(
    seen,
    @nospecialize(prev::Oceananigans.Grids.OrthogonalSphericalShellGrid),
    args...;
    kwargs...
    ) = Reactant.make_tracer_via_immutable_constructor(seen, prev, args...; kwargs...)

# https://github.com/CliMA/Oceananigans.jl/blob/d9b3b142d8252e8e11382d1b3118ac2a092b38a2/src/ImmersedBoundaries/immersed_boundary_grid.jl#L8
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, G, I, M, S, Arch}
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    S2 = Reactant.traced_type_inner(S, seen, mode, track_numbers, sharding, runtime)
    FT2 = eltype(G2)
    return Oceananigans.Grids.ImmersedBoundaryGrid{FT2, TX2, TY2, TZ2, G2, I2, M2, S2, Arch}
end

struct Fix1v2{F,T}
    f::F
    t::T
end

@inline function (s::Fix1v2)(args...)
    s.f(s.t, args...)
end

function evalcond(c, i, j, k)
    Oceananigans.AbstractOperations.evaluate_condition(c.condition, i, j, k, c.grid, c)
end

@inline function Reactant.TracedUtils.broadcast_to_size(c::Oceananigans.AbstractOperations.ConditionalOperation, rsize)
    if c == rsize
        return Reactant.TracedUtils.materialize_traced_array(c)
    end
    return c
end

@inline function Reactant.TracedUtils.materialize_traced_array(c::Oceananigans.AbstractOperations.ConditionalOperation)
    N = ndims(c)
    axes2 = ntuple(Val(N)) do i
        reshape(Base.OneTo(size(c, i)), (ntuple(Val(N)) do j
            if i == j
                size(c, i)
            else
                1
            end
        end)...)
    end
    tracedidxs = axes2

    conds = Reactant.TracedUtils.materialize_traced_array(Reactant.call_with_reactant(Oceananigans.AbstractOperations.evaluate_condition, c.condition, tracedidxs..., c.grid, c))

    @assert size(conds) == size(c)
    tvals = Reactant.Ops.fill(zero(Reactant.unwrapped_eltype(Base.eltype(c))), size(c))

    @assert size(tvals) == size(c)
    gf =  Reactant.call_with_reactant(getindex, c.operand, axes2...)
    Reactant.TracedRArrayOverrides._copyto!(tvals, Base.broadcasted(c.func, gf))

    return Reactant.Ops.select(
                conds,
                tvals,
                Reactant.TracedUtils.broadcast_to_size(c.mask, size(c))
    )
end

function evalkern(kern, i, j, k)
    kern.kernel_function(i, j, k, kern.grid, kern.arguments...)
end

@inline function Reactant.TracedUtils.materialize_traced_array(c::Oceananigans.AbstractOperations.KernelFunctionOperation)
    N = ndims(c)
    axes2 = ntuple(Val(N)) do i
        reshape(Base.OneTo(size(c, i)), (ntuple(Val(N)) do j
            if i == j
                size(c, i)
            else
                1
            end
        end)...)
    end

    tvals = Reactant.Ops.fill(Reactant.unwrapped_eltype(Base.eltype(c)), size(c))
    Reactant.TracedRArrayOverrides._copyto!(tvals, Base.broadcasted(Fix1v2(evalkern, c), axes2...))
    return tvals
end

function Oceananigans.TimeSteppers.tick_time!(clock::Oceananigans.TimeSteppers.Clock{<:Reactant.TracedRNumber}, Δt)
    nt = Oceananigans.TimeSteppers.next_time(clock, Δt)
    clock.time.mlir_data = nt.mlir_data
    nt
end

function Oceananigans.TimeSteppers.tick!(clock::Oceananigans.TimeSteppers.Clock{<:Any, <:Any, <:Reactant.TracedRNumber}, Δt; stage=false)
    Oceananigans.TimeSteppers.tick_time!(clock, Δt)

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration.mlir_data = (clock.iteration + 1).mlir_data
        clock.stage = 1
    end

    return nothing
end

@inline function Reactant.TracedUtils.broadcast_to_size(c::Oceananigans.AbstractOperations.KernelFunctionOperation, rsize)
    if c == rsize
        return Reactant.TracedUtils.materialize_traced_array(c)
    end
    return c
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


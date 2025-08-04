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
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.ConditionalOperation{LX, LY, LZ, F, C, O, G, M, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, F, C, O, G, M, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)
    F2 = Reactant.traced_type_inner(F, seen, mode, track_numbers, sharding, runtime)
    C2 = Reactant.traced_type_inner(C, seen, mode, track_numbers, sharding, runtime)
    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(O2)
    return Oceananigans.AbstractOperations.ConditionalOperation{LX2, LY2, LZ2, F2, C2, O2, G2, M2, T2}
end

# https://github.com/CliMA/Oceananigans.jl/blob/c29939097a8d2f42966e930f2f2605803bf5d44c/src/AbstractOperations/binary_operations.jl#L5
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, G, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, O, A, B, IA, IB, G, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)

    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)

    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, sharding, runtime)
    B2 = Reactant.traced_type_inner(B, seen, mode, track_numbers, sharding, runtime)
    IA2 = Reactant.traced_type_inner(IA, seen, mode, track_numbers, sharding, runtime)
    IB2 = Reactant.traced_type_inner(IB, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)

    T2 = eltype(G2)
    return Oceananigans.AbstractOperations.BinaryOperation{LX2, LY2, LZ2, O2, A2, B2, IA2, IB2, G2, T2}
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
    for NF in (CC2, FC2, CF2, FF2)
	if NF === Nothing
	   continue
	end
	FT2 = Reactant.promote_traced_type(FT2, eltype(NF))
    end
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

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{LatitudeLongitudeGrid{FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC, 
                                                 DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC, DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I} 
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    Z2 = Reactant.traced_type_inner(Z, seen, mode, track_numbers, sharding, runtime)
    DXF2 = Reactant.traced_type_inner(DXF, seen, mode, track_numbers, sharding, runtime)
    DXC2 = Reactant.traced_type_inner(DXC, seen, mode, track_numbers, sharding, runtime)
    XF2 = Reactant.traced_type_inner(XF, seen, mode, track_numbers, sharding, runtime)
    XC2 = Reactant.traced_type_inner(XC, seen, mode, track_numbers, sharding, runtime)
    DYF2 = Reactant.traced_type_inner(DYF, seen, mode, track_numbers, sharding, runtime)
    DYC2 = Reactant.traced_type_inner(DYC, seen, mode, track_numbers, sharding, runtime)
    YF2 = Reactant.traced_type_inner(YF, seen, mode, track_numbers, sharding, runtime)
    YC2 = Reactant.traced_type_inner(YC, seen, mode, track_numbers, sharding, runtime)
    DXCC2 = Reactant.traced_type_inner(DXCC, seen, mode, track_numbers, sharding, runtime)
    DXFC2 = Reactant.traced_type_inner(DXFC, seen, mode, track_numbers, sharding, runtime)
    DXCF2 = Reactant.traced_type_inner(DXCF, seen, mode, track_numbers, sharding, runtime)
    DXFF2 = Reactant.traced_type_inner(DXFF, seen, mode, track_numbers, sharding, runtime)
    DYFC2 = Reactant.traced_type_inner(DYFC, seen, mode, track_numbers, sharding, runtime)
    DYCF2 = Reactant.traced_type_inner(DYCF, seen, mode, track_numbers, sharding, runtime)
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)

    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)

    for NF in (XF2, XC2, YF2, YC2, DXCC2, DXFC2, DYCF2, DYCF2, DXFF2)
	if NF === Nothing
	   continue
	end
	FT2 = Reactant.promote_traced_type(FT2, eltype(NF))
    end

    res = Oceananigans.Grids.LatitudeLongitudeGrid{FT2, TX2, TY2, TZ2, Z2, DXF2, DXC2, XF2, XC2, DYF2, DYC2, YF2, YC2, 
                                                 DXCC2, DXFC2, DXCF2, DXFF2, DYFC2, DYCF2, Arch, I2}
    return res
end

@inline Reactant.make_tracer(
    seen,
    @nospecialize(prev::Oceananigans.Grids.LatitudeLongitudeGrid),
    args...;
    kwargs...
    ) = Reactant.make_tracer_via_immutable_constructor(seen, prev, args...; kwargs...)

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

    conds = Reactant.call_with_reactant(Oceananigans.AbstractOperations.evaluate_condition, c.condition, tracedidxs..., c.grid, c)
    if conds isa Bool
      conds = Reactant.Ops.fill(conds, size(c))
    else
      conds = Reactant.TracedUtils.materialize_traced_array(conds)
    end

    @assert size(conds) == size(c)
    tvals = Reactant.Ops.fill(zero(Reactant.unwrapped_eltype(Base.eltype(c))), size(c))

    @assert size(tvals) == size(c)
    gf =  Reactant.call_with_reactant(getindex, c.operand, axes2...)
    if gf isa AbstractFloat
	 gf = Reactant.Ops.fill(gf, size(c))
    end
    Reactant.TracedRArrayOverrides._copyto!(tvals, Base.broadcasted(c.func isa Nothing ? Base.identity : c.func, gf))

    mask = c.mask
    if mask isa AbstractFloat && typeof(mask) != Reactant.unwrapped_eltype(Base.eltype(c))
	mask = Base.eltype(c)(mask)
    end

    return Reactant.Ops.select(
                conds,
                tvals,
                Reactant.TracedUtils.broadcast_to_size(mask, size(c))
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
        clock.last_stage_Δt = Δt
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


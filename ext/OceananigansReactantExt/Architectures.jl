module Architectures

using Reactant
using Oceananigans
using Oceananigans.DistributedComputations: Distributed

using Reactant: AnyConcreteRArray

import Oceananigans.Architectures: device, architecture, array_type, on_architecture
import Oceananigans.Architectures: unified_array, ReactantState, device_copy_to!

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend
const AnyConcreteReactantArray = Union{Reactant.AnyConcretePJRTArray,Reactant.AnyConcreteIFRTArray}

device(::ReactantState) = ReactantBackend()

architecture(::AnyConcreteReactantArray) = ReactantState()
architecture(::Reactant.AnyTracedRArray) = ReactantState()

# ConcreteRArray can refer to either a PJRT or IFRT array based on Reactant preferences
array_type(::ReactantState) = ConcreteRArray

on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
on_architecture(::CPU, a::AnyConcreteReactantArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any,<:Any,<:AnyConcreteReactantArray}) = Array(a)

const ArraysToRArray = Union{Array,
    Reactant.AnyConcretePJRTArray,
    # Reactant.AnyConcreteIFRTArray, # needed?
    BitArray,
    SubArray{<:Any,<:Any,<:Array}}

on_architecture(::ReactantState, a::ArraysToRArray) = Reactant.to_rarray(a)
Oceananigans.Architectures.cpu_architecture(arch::Distributed{<:ReactantState}) = CPU()

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::AnyConcreteReactantArray, src::AnyConcreteReactantArray; kw...) = Base.copyto!(dst, src)

# Distributed computations
function Oceananigans.Distributed(arch::ReactantState; devices=nothing,
    partition=nothing, kw...)
    if devices === nothing
        devices = Reactant.devices()

        if any(!Reactant.XLA.is_addressable, devices)
            # This means we are using a distributed environment
            if !Reactant.Distributed.is_initialized()
                # Try to setup the distributed environment. This will automatically
                # fail if not possible
                Reactant.Distributed.initialize()
            end
            devices = Reactant.devices()
        end
    end

    if partition === nothing
        partition = Oceananigans.Partition(length(devices))
    end

    ranks = Rx, Ry, Rz = size(partition)
    partition_ranks = Rx * Ry * Rz

    if partition_ranks < length(devices)
        @warn "Only using $partition_ranks of $(length(devices)) devices"
        devices = devices[1:partition_ranks]
    end

    if partition_ranks != length(devices)
        throw(ArgumentError("Partition($Rx, $Ry, $Rz) [$partition_ranks ranks] \
                             inconsistent with $(length(devices)) devices"))
    end

    local_rank = Reactant.Distributed.local_rank()
    local_index = Oceananigans.DistributedComputations.rank2index(local_rank, Rx, Ry, Rz)

    mesh = Sharding.Mesh(
        reshape(devices, size(partition)...),
        (:x, :y, :z),
    )

    # Sharding does not need all the infrastructure for pipelining
    # since it should handle and optimize communication by itself
    synchronized_communcation = true

    return Oceananigans.Distributed{synchronized_communcation}(arch, partition, ranks, local_rank, local_index,
                                                               mesh, nothing, nothing, Ref(0), devices)
end 

Oceananigans.Grids.unwrapped_eltype(T::Type{<:Reactant.ConcretePJRTNumber}) = Reactant.unwrapped_eltype(T)
Oceananigans.Grids.unwrapped_eltype(T::Type{<:Reactant.ConcreteIFRTNumber}) = Reactant.unwrapped_eltype(T)

end # module

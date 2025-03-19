module Architectures

using Reactant
using Oceananigans

using Reactant: AnyConcreteRArray

import Oceananigans.Architectures: device, architecture, array_type, on_architecture
import Oceananigans.Architectures: unified_array, ReactantState, device_copy_to!

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend

const AnyConcreteReactantArray = Union{Reactant.AnyConcretePJRTArray,Reactant.AnyConcreteIFRTArray}

device(::ReactantState) = ReactantBackend()

architecture(::AnyConcreteReactantArray) = ReactantState
architecture(::Reactant.AnyTracedRArray) = ReactantState

# ConcreteRArray can refer to either a PJRT or IFRT array based on Reactant preferences
array_type(::ReactantState) = ConcreteRArray

on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
on_architecture(::CPU, a::AnyConcreteReactantArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any,<:Any,<:AnyConcreteReactantArray}) = Array(a)

const ArraysToRArray = Union{Array,
    Reactant.AnyConcretePJRTArray,
    BitArray,
    SubArray{<:Any,<:Any,<:Array}}

on_architecture(::ReactantState, a::ArraysToRArray) = Reactant.to_rarray(a)

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::AnyConcreteReactantArray, src::AnyConcreteReactantArray; kw...) = Base.copyto!(dst, src)

# Distributed computations
function Oceananigans.Distributed(arch::ReactantState; devices=nothing,
    partition=nothing, kw...)
    if devices === nothing
        # TODO: Can be made better
        if Reactant.Distributed.initialized[]
            devices = Reactant.devices()
        else
            devices = Reactant.addressable_devices()
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

    # XXX: Internal API
    local_rank = Reactant.XLA.global_state.process_id
    local_index = Oceananigans.DistributedComputations.rank2index(local_rank, Rx, Ry, Rz)

    mesh = Sharding.Mesh(
        reshape(devices, size(partition)...),
        (:x, :y, :z),
    )

    # TODO: have a different field for mesh in Distributed
    return Oceananigans.Distributed{false}(arch, partition, ranks, local_rank, local_index,
        mesh, nothing, nothing, Ref(0), devices)
end

end # module

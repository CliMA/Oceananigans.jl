module Architectures

using Reactant
using Oceananigans
using Oceananigans.DistributedComputations: Distributed

using Reactant: AnyConcreteRArray

import Oceananigans.Grids: LatitudeLongitudeGrid, topology
import Oceananigans.Architectures: device, architecture, array_type, on_architecture, child_architecture
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
    total_ranks = Rx * Ry * Rz

    if total_ranks != length(devices)
        throw(ArgumentError("Partition($Rx, $Ry, $Rz) [$total_ranks ranks] \
                             inconsistent with $(length(devices)) devices"))
    end

    local_rank = Reactant.Distributed.local_rank()
    local_index = Oceananigans.DistributedComputations.rank2index(local_rank, Rx, Ry, Rz)

    partition_size = size(partition)
    @assert length(partition_size) == 3
    if partition_size[3] == 1
        mesh = Sharding.Mesh(
            reshape(devices, partition_size[1], partition_size[2]), (:x, :y),
        )
    else
        mesh = Sharding.Mesh(reshape(devices, partition_size...), (:x, :y, :z))
    end

    # Syncronized communication does not mean anything in this case so we set it to nothing
    return Oceananigans.Distributed{nothing}(arch, partition, ranks, local_rank, local_index,
                                             mesh, nothing, nothing, Ref(0), devices)
end

Oceananigans.Grids.unwrapped_eltype(T::Type{<:Reactant.ConcretePJRTNumber}) = Reactant.unwrapped_eltype(T)
Oceananigans.Grids.unwrapped_eltype(T::Type{<:Reactant.ConcreteIFRTNumber}) = Reactant.unwrapped_eltype(T)


function on_architecture(new_arch::Distributed{<:ReactantState}, old_grid::LatitudeLongitudeGrid)
    child_arch = child_architecture(new_arch)
    old_properties = (old_grid.Δλᶠᵃᵃ, old_grid.Δλᶜᵃᵃ, old_grid.λᶠᵃᵃ,  old_grid.λᶜᵃᵃ,
                      old_grid.Δφᵃᶠᵃ, old_grid.Δφᵃᶜᵃ, old_grid.φᵃᶠᵃ,  old_grid.φᵃᶜᵃ,
                      old_grid.z,
                      old_grid.Δxᶠᶜᵃ, old_grid.Δxᶜᶠᵃ, old_grid.Δxᶠᶠᵃ, old_grid.Δxᶜᶜᵃ,
                      old_grid.Δyᶠᶜᵃ, old_grid.Δyᶜᶠᵃ,
                      old_grid.Azᶠᶜᵃ, old_grid.Azᶜᶠᵃ, old_grid.Azᶠᶠᵃ, old_grid.Azᶜᶜᵃ)

    sharding = Sharding.DimsSharding(new_arch.connectivity, (1, 2), (:x, :y))
    new_properties = Tuple(Reactant.to_rarray(p; sharding) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return LatitudeLongitudeGrid{TX, TY, TZ}(new_arch,
                                             old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                             old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                             old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                             new_properties...,
                                             old_grid.radius)
end

end # module

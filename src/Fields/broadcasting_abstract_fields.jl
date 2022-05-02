#####
##### Broadcasting utilities
#####

using Base.Broadcast: DefaultArrayStyle
using Base.Broadcast: Broadcasted
using CUDA

using Oceananigans.Architectures: device_event

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

Base.Broadcast.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()

# Precedence rule
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::DefaultArrayStyle{N}) where N = FieldBroadcastStyle()
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::CUDA.CuArrayStyle{N}) where N = FieldBroadcastStyle()

# For use in Base.copy when broadcasting with numbers and arrays (useful for comparisons like f::AbstractField .== 0)
Base.similar(bc::Broadcasted{FieldBroadcastStyle}, ::Type{ElType}) where ElType = similar(Array{ElType}, axes(bc))

# Bypass style combining for in-place broadcasting with arrays / scalars to use built-in broadcasting machinery
const BroadcastedArrayOrCuArray = Union{Broadcasted{<:DefaultArrayStyle},
                                        Broadcasted{<:CUDA.CuArrayStyle}}

@inline Base.Broadcast.materialize!(dest::AbstractField, bc::BroadcastedArrayOrCuArray) =
    Base.Broadcast.materialize!(interior(dest), bc)

# TODO: make this support Field that are windowed in _only_ 1 or 2 dimensions.
# Right now, this may only produce expected behavior (re: dimensionality) for
# WindowedField that are windowed in three-dimensions. Of course, broadcasting with
# scalar `bc` is no issue.
@inline Base.Broadcast.materialize!(dest::WindowedField, bc::BroadcastedArrayOrCuArray) =
    Base.Broadcast.materialize!(parent(dest), bc)

#####
##### Kernels
#####

@inline offset_compute_index(::Colon, i) = i
@inline offset_compute_index(range::UnitRange, i) = range[1] + i - 1

@kernel function broadcast_kernel!(dest, bc, index_ranges)
    i, j, k = @index(Global, NTuple)

    i′ = offset_compute_index(index_ranges[1], i)
    j′ = offset_compute_index(index_ranges[2], j)
    k′ = offset_compute_index(index_ranges[3], k)

    @inbounds dest[i′, j′, k′] = bc[i′, j′, k′]
end

# Interface for getting AbstractOperation right
broadcasted_to_abstract_operation(loc, grid, a) = a

# Broadcasting with interpolation breaks Base's default rules,
# so we bypass the infrastructure for checking axes compatibility,
# and head straight to copyto! from materialize!.
@inline Base.Broadcast.materialize!(::Base.Broadcast.BroadcastStyle,
                                    dest::AbstractField,
                                    bc::Broadcasted{<:FieldBroadcastStyle}) = copyto!(dest, convert(Broadcasted{Nothing}, bc))

@inline function Base.copyto!(dest::Field, bc::Broadcasted{Nothing})

    grid = dest.grid
    arch = architecture(dest)

    bc′ = broadcasted_to_abstract_operation(location(dest), grid, bc)

    event = launch!(arch, grid, size(dest), broadcast_kernel!, dest, bc′, dest.indices)
    wait(device(arch), event)

    return dest
end


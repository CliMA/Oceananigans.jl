#####
##### Broadcasting utilities
#####

using Base.Broadcast: DefaultArrayStyle
using Base.Broadcast: Broadcasted
using CUDA
using AMDGPU

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

Base.Broadcast.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()

# Precedence rule
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::DefaultArrayStyle{N}) where N = FieldBroadcastStyle()
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::CUDA.CuArrayStyle{N}) where N = FieldBroadcastStyle()
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::AMDGPU.ROCArrayStyle{N}) where N = FieldBroadcastStyle()

# For use in Base.copy when broadcasting with numbers and arrays (useful for comparisons like f::AbstractField .== 0)
Base.similar(bc::Broadcasted{FieldBroadcastStyle}, ::Type{ElType}) where ElType = similar(Array{ElType}, axes(bc))

# Bypass style combining for in-place broadcasting with arrays / scalars to use built-in broadcasting machinery
const BroadcastedArrayOrCuArray = Union{Broadcasted{<:DefaultArrayStyle},
                                        Broadcasted{<:CUDA.CuArrayStyle},
                                        Broadcasted{<:AMDGPU.ROCArrayStyle}}

@inline function Base.Broadcast.materialize!(dest::Field, bc::BroadcastedArrayOrCuArray)
    if any(a isa OffsetArray for a in bc.args)
        return Base.Broadcast.materialize!(dest.data, bc)
    else
        return Base.Broadcast.materialize!(interior(dest), bc)
    end
end

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

@inline offset_index(::Colon) = 0
@inline offset_index(range::UnitRange) = range[1] - 1

@kernel function _broadcast_kernel!(dest, bc)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = bc[i, j, k]
end

# Interface for getting AbstractOperation right
@inline broadcasted_to_abstract_operation(loc, grid, a) = a

# Broadcasting with interpolation breaks Base's default rules,
# so we bypass the infrastructure for checking axes compatibility,
# and head straight to copyto! from materialize!.
@inline function Base.Broadcast.materialize!(::Base.Broadcast.BroadcastStyle,
                                             dest::Field,
                                             bc::Broadcasted{<:FieldBroadcastStyle})

    return copyto!(dest, convert(Broadcasted{Nothing}, bc))
end

@inline function Base.copyto!(dest::Field, bc::Broadcasted{Nothing})

    grid = dest.grid
    arch = architecture(dest)
    bc′ = broadcasted_to_abstract_operation(location(dest), grid, bc)

    param = KernelParameters(size(dest), map(offset_index, dest.indices))
    launch!(arch, grid, param, _broadcast_kernel!, dest, bc′)

    return dest
end

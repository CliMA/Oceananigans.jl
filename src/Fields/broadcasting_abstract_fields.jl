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

#####
##### Kernels
#####

@kernel function broadcast_kernel!(dest, bc)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = bc[i, j, k]
end

broadcasted_to_abstract_operation(loc, grid, a) = a

# Broadcasting with interpolation breaks Base's default rules for AbstractOperations 
@inline Base.Broadcast.materialize!(::Base.Broadcast.BroadcastStyle,
                                    dest::AbstractField,
                                    bc::Broadcasted{<:FieldBroadcastStyle}) = copyto!(dest, convert(Broadcasted{Nothing}, bc))

@inline function Base.copyto!(dest::Field, bc::Broadcasted{Nothing})

    grid = dest.grid
    arch = architecture(dest)

    bc′ = broadcasted_to_abstract_operation(location(dest), grid, bc)

    event = launch!(arch, grid, :xyz, broadcast_kernel!, dest, bc′,
                    include_right_boundaries = true,
                    dependencies = device_event(arch),
                    reduced_dimensions = reduced_dimensions(dest),
                    location = location(dest))

    wait(device(arch), event)

    return dest
end


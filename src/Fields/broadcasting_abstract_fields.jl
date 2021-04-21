#####
##### Broadcasting utilities
#####

using Base.Broadcast: DefaultArrayStyle
using Base.Broadcast: Broadcasted

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

Base.Broadcast.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()

# Precedence rule
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::DefaultArrayStyle{N}) where N = FieldBroadcastStyle()

# For use in Base.copy when broadcasting with numbers and arrays (useful for comparisons like f::AbstractField .== 0)
Base.similar(bc::Broadcasted{FieldBroadcastStyle}, ::Type{ElType}) where ElType = similar(Array{ElType}, axes(bc))

# Bypass style combining for in-place broadcasting with arrays / scalars to use built-in broadcasting machinery
@inline Base.Broadcast.materialize!(dest::AbstractField, bc::Broadcasted{<:DefaultArrayStyle}) =
    Base.Broadcast.materialize!(DefaultArrayStyle{3}(), dest, bc)

#####
##### Kernels
#####

@kernel function broadcast_xyz!(dest, bc)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = bc[i, j, k]
end

@kernel function broadcast_xy!(dest, bc)
    i, j = @index(Global, NTuple)
    @inbounds dest[i, j, 1] = bc[i, j, 1]
end

@kernel function broadcast_xz!(dest, bc)
    i, k = @index(Global, NTuple)
    @inbounds dest[i, 1, k] = bc[i, 1, k]
end

@kernel function broadcast_yz!(dest, bc)
    j, k = @index(Global, NTuple)
    @inbounds dest[1, j, k] = bc[1, j, k]
end

# Three-dimensional general case

launch_configuration(::AbstractField) = :xyz
broadcast_kernel(::AbstractField) = broadcast_xyz!

# Two dimensional reduced field

const Loc = Union{Center, Face}

broadcast_kernel(::AbstractReducedField{Nothing, <:Loc, <:Loc}) = broadcast_yz!
broadcast_kernel(::AbstractReducedField{<:Loc, Nothing, <:Loc}) = broadcast_xz!
broadcast_kernel(::AbstractReducedField{<:Loc, <:Loc, Nothing}) = broadcast_xy!

launch_configuration(::AbstractReducedField{Nothing, <:Loc, <:Loc}) = :yz
launch_configuration(::AbstractReducedField{<:Loc, Nothing, <:Loc}) = :xz
launch_configuration(::AbstractReducedField{<:Loc, <:Loc, Nothing}) = :xy

broadcasted_to_abstract_operation(loc, grid, a) = a

# Broadcasting with interpolation breaks Base's default rules for AbstractOperations 
@inline Base.Broadcast.materialize!(::Base.Broadcast.BroadcastStyle,
                                    dest::AbstractField,
                                    bc::Broadcasted{<:FieldBroadcastStyle}) = copyto!(dest, convert(Broadcasted{Nothing}, bc))

@inline function Base.copyto!(dest::AbstractField{X, Y, Z}, bc::Broadcasted{Nothing}) where {X, Y, Z}

    grid = dest.grid
    arch = architecture(dest)
    config = launch_configuration(dest)
    kernel = broadcast_kernel(dest)

    bc′ = broadcasted_to_abstract_operation(location(dest), grid, bc)

    event = launch!(arch, grid, config, kernel, dest, bc′,
                    include_right_boundaries = true,
                    location = (X, Y, Z))

    wait(device(arch), event)

    fill_halo_regions!(dest, arch)

    return nothing
end

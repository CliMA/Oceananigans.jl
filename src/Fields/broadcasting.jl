#####
##### Broadcasting utilities
#####

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

Base.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()

using Base.Broadcast: Broadcasted

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

@inline function Base.copyto!(dest::AbstractField{X, Y, Z}, bc::Broadcasted{Nothing}) where {X, Y, Z}

    # Is this needed?
    bc′ = Broadcast.preprocess(dest, bc)

    grid = dest.grid
    arch = architecture(dest)
    config = launch_configuration(dest)
    kernel = broadcast_kernel(dest)

    event = launch!(arch, grid, config, kernel, dest, bc′,
                    include_right_boundaries = true,
                    location = (X, Y, Z))

    wait(device(arch), event)

    return nothing
end

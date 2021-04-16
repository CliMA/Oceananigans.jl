#####
##### Broadcasting utilities
#####

using Base.Broadcast: Broadcasted

@kernel function broadcast_3d!(dest, bc)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = bc[i, j, k]
end

@kernel function broadcast_2d!(dest, bc)
    i, j = @index(Global, NTuple)
    @inbounds dest[i, j] = bc[i, j]
end

# Three-dimensional general case

launch_configuration(::AbstractField) = :xyz
broadcast_kernel(::AbstractField) = broadcast_3d!

# Two dimensional reduced field

const Loc = Union{Center, Face}

const TwoDimensionalReducedField = Union{AbstractReducedField{Nothing, <:Loc, <:Loc},
                                         AbstractReducedField{<:Loc, Nothing, <:Loc},
                                         AbstractReducedField{<:Loc, <:Loc, Nothing}}

broadcast_kernel(::TwoDimensionalReducedField) = broadcast_2d!

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

#####
##### Broadcasting utilities
#####

using Base.Broadcast: DefaultArrayStyle
using Base.Broadcast: Broadcasted

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

# This code defines BroadcastStyle for AbstractField and associated paraphernalia.
# It causes problems for simple cases so it's commented out right now.
# However, we need to use this machinery to ensure proper interpolation between
# fields at different locations on the staggered grid, so we keep it here, commented out,
# in the hope that it will be employed someday.

Base.Broadcast.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()
Base.Broadcast.BroadcastStyle(::Type{<:AbstractReducedField}) = FieldBroadcastStyle()
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, b::DefaultArrayStyle{N}) where N = DefaultArrayStyle{3}()

"`A = find_field(As)` returns the first AbstractField among the arguments."
find_field(bc::Broadcasted) = find_field(bc.args)
find_field(args::Tuple) = find_field(find_field(args[1]), Base.tail(args))
find_field(x) = x
find_field(::Tuple{}) = nothing
find_field(a::AbstractField, rest) = a
find_field(::Any, rest) = find_field(rest)

Base.similar(bc::Broadcasted{FieldBroadcastStyle}, ::Type{ElType}) where ElType = similar(Array{ElType}, axes(bc))

function Base.similar(bc::Broadcasted{FieldBroadcastStyle}, ::Type{<:AbstractFloat})
    field = find_field(bc)
    return similar(field)
end

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

# Insert locations into AbstractOperations embedded in Broadcasted trees
insert_destination_location(loc, bc::Broadcasted) where S = Broadcasted{S}(bc.f, insert_destination_location(loc, bc.args), bc.axes)
insert_destination_location(loc, args::Tuple) = Tuple(insert_destination_location(loc, a) for a in args)

broadcasted_to_abstract_operation(loc, grid, a) = a

broadcasted_to_abstract_operation(loc, grid, bc::Broadcasted{<:Any, <:Any, <:Any, <:Any}) =
    bc.f(loc, Tuple(broadcasted_to_abstract_operation(loc, grid, a) for a in bc.args)...)

@inline function Base.copyto!(dest::AbstractField{X, Y, Z}, bc::Broadcasted{Nothing}) where {X, Y, Z}

    # Is this needed?
    # bc′ = Broadcast.preprocess(dest, bc)

    grid = dest.grid
    arch = architecture(dest)
    config = launch_configuration(dest)
    kernel = broadcast_kernel(dest)

    # Convert Broadcasted to AbstractOperation
    op = broadcasted_to_abstract_operation(location(dest), grid, bc)

    event = launch!(arch, grid, config, kernel, dest, op,
                    include_right_boundaries = true,
                    location = (X, Y, Z))

    wait(device(arch), event)

    fill_halo_regions!(dest, arch)

    return nothing
end

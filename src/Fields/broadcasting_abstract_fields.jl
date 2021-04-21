#####
##### Broadcasting utilities
#####

using Base.Broadcast: DefaultArrayStyle
using Base.Broadcast: Broadcasted

struct FieldBroadcastStyle <: Broadcast.AbstractArrayStyle{3} end

# The below needs to be uncommented when FieldBroadcastStyle shenanigans
# and result array inference for allocating broadcast operations is figured out.
Base.Broadcast.BroadcastStyle(::Type{<:AbstractField}) = FieldBroadcastStyle()

# Precedence rules
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::DefaultArrayStyle{N}) where N = DefaultArrayStyle{3}()
Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::FieldBroadcastStyle) = FieldBroadcastStyle()

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

broadcasted_to_abstract_operation(loc, grid, bc::Broadcasted{<:Any, <:Any, <:Any, <:Any}) =
    bc.f(loc, Tuple(broadcasted_to_abstract_operation(loc, grid, a) for a in bc.args)...)

#####
##### Hopefully we don't need to interpolate, but if we do...
#####

needs_interpolation(Xa, Xb) = false
needs_interpolation(::Type{Center}, ::Type{Face}) = true
needs_interpolation(::Type{Face}, ::Type{Center}) = true

needs_interpolation(La::Tuple, Lb::Tuple) = any(needs_interpolation.(La, Lb))
needs_interpolation(La::Tuple, b::AbstractField) = needs_interpolation(La, location(b))
needs_interpolation(La::Tuple, ::Number) = false
needs_interpolation(La::Tuple, ::AbstractArray) = false
needs_interpolation(La::Tuple, bc::Broadcasted) = any(needs_interpolation(La, b) for b in bc.args)

# Broadcasting with interpolation breaks Base's default rules for ABroBbstractOperations 
@inline Base.Broadcast.materialize!(::BroadcastStyle, dest::AbstractField, bc::Broadcasted) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))

# Non-interpolated broadcasting to Field
@inline function Base.copyto!(dest::AbstractField{X, Y, Z}, bc::Broadcasted{Nothing}) where {X, Y, Z}

    # It's probably best if you don't need interpolation, but anyways...
    #bc′ = needs_interpolation(dest, bc) ? broadcasted_to_abstract_operation(location(dest), grid, bc) :
    #                                      Base.Broadcast.preprocess(dest, bc)
                                            
    grid = dest.grid
    arch = architecture(dest)
    config = launch_configuration(dest)
    kernel = broadcast_kernel(dest)

    @show bc′ = broadcasted_to_abstract_operation(location(dest), grid, bc)

    event = launch!(arch, grid, config, kernel, dest, bc′,
                    include_right_boundaries = true,
                    location = (X, Y, Z))

    wait(device(arch), event)

    fill_halo_regions!(dest, arch)

    return nothing
end

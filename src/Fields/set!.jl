using KernelAbstractions: @kernel, @index

using Oceananigans.Grids: node_names
using Oceananigans.Architectures: cpu_architecture, GPU, CPU, ReactantState

#####
##### Utilities
#####

function tuple_string(tup::Tuple)
    str = prod(string(t, ", ") for t in tup)
    return str[1:end-2] # remove trailing ", "
end

tuple_string(tup::Tuple{}) = ""

#####
##### set!
#####

set!(obj::AbstractField, ::Nothing) = nothing

function set!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set!(ϕ, value)
    end
    return nothing
end

function set!(ft::NamedFieldTuple, a::Number)
    for field in ft
        set!(field, a)
    end
    return ft
end

function set!(dst::NamedFieldTuple, src::NamedTuple)
    for name in keys(dst)
        set!(dst[name], src[name])
    end
    return dst
end

# This interface helps us do things like set distributed fields
set!(u::Field, f::Function) = set_to_function!(u, f)
set!(u::Field, a::Union{Array, OffsetArray}) = set_to_array!(u, a)

"""
$(TYPEDSIGNATURES)

Set `u` from `v`. When `u` and `v` have the same `size`, `location`, and
`indices`, the data of `v` is copied into `u` (cross-architecture transfers
are handled automatically). Otherwise, `v` is migrated to `u`'s architecture
if needed, its halo regions are filled, and then it is interpolated onto `u`
with [`interpolate!`](@ref). This means field-to-field `set!` "just works"
across grids of different resolution, between staggered locations, and across
architectures.

Note that the interpolation path samples `v` pointwise; for conservative
remapping, call [`regrid!`](@ref) explicitly.
"""
set!(u::Field, v::Field) = set_to_field!(u, v)

function set!(u::Field, a::Number)
    fill!(interior(u), a) # note all other set! only change interior
    return u # return u, not parent(u), for type-stability
end

function set!(u::Field, v)
    u .= v # fallback
    return u
end

set!(u::Field, z::ZeroField) = set!(u, zero(eltype(u)))

#####
##### Setting to specific things
#####

function set_to_function!(u, f, clock=nothing)
    # Supports serial and distributed
    arch = architecture(u)
    child_arch = child_architecture(u)

    # Determine cpu_grid and cpu_u
    cpu_grid, cpu_u = if child_arch isa GPU || child_arch isa ReactantState
        cpu_arch = cpu_architecture(arch)
        cpu_grid = on_architecture(cpu_arch, u.grid)
        cpu_grid, Field(instantiated_location(u), cpu_grid; indices = indices(u))
    elseif child_arch isa CPU
        u.grid, u
    end

    # Form a FunctionField from `f`
    LX, LY, LZ = location(u)
    f_field = FunctionField{LX, LY, LZ}(f, cpu_grid; clock)

    # Try to set the FunctionField to cpu_u
    try
        set!(cpu_u, f_field)
    catch err
        u_loc = Tuple(L() for L in location(u))

        arg_str  = tuple_string(node_names(u.grid, u_loc...))
        loc_str  = tuple_string(location(u))
        topo_str = tuple_string(topology(u.grid))

        msg = string("An error was encountered within set! while setting the field", '\n', '\n',
                     "    ", prettysummary(u), '\n', '\n',
                     "Note that to use set!(field, func::Function) on a field at location ",
                     "(", loc_str, ")", '\n',
                     "and on a grid with topology (", topo_str, "), func must be ",
                     "callable via", '\n', '\n',
                     "     func(", arg_str, ")", '\n')
        @warn msg
        throw(err)
    end

    # Transfer data to GPU if u is on the GPU
    if child_arch isa GPU || child_arch isa ReactantState
        set!(u, cpu_u)
    end
    return u
end

function set_to_array!(u, a)
    a = on_architecture(architecture(u), a)

    try
        copyto!(u, a)
    catch err
        if err isa DimensionMismatch
            Nx, Ny, Nz = size(u)
            u .= reshape(a, Nx, Ny, Nz)

            msg = string("Reshaped ", summary(a),
                         " to set! its data to ", '\n',
                         summary(u))
            @warn msg
        else
            throw(err)
        end
    end

    return u
end

function set_to_field!(u, v)
    if copyable_fields(u, v)
        copy_to_field!(u, v)
    else
        # Fill halos on v's native architecture so distributed dispatch (if any) is used;
        # on_architecture would strip Distributed{CPU} to CPU while keeping distributed
        # boundary conditions, mismatching fill_halo_regions! dispatch.
        fill_halo_regions!(v)
        v_on_u = on_architecture(child_architecture(u), v)
        interpolate!(u, v_on_u)
    end

    return u
end

# `u` may be filled from `v` by copying, rather than by interpolating, when every
# dimension is copyable. Note that sizes need not match: `v` may be reduced.
function copyable_fields(u, v)
    ℓu = location(u)
    ℓv = location(v)
    iu = indices(u)
    iv = indices(v)
    Nu = size(u)
    Nv = size(v)
    return copyable_dimension(ℓu[1], ℓv[1], iu[1], iv[1], Nu[1], Nv[1]) &&
           copyable_dimension(ℓu[2], ℓv[2], iu[2], iv[2], Nu[2], Nv[2]) &&
           copyable_dimension(ℓu[3], ℓv[3], iu[3], iv[3], Nu[3], Nv[3])
end

# Along a dimension, `u` may be filled from `v` by copying in three mutually exclusive
# situations: the two fields are discretized equivalently, the dimension is degenerate,
# or `v` is reduced and stretches across `u`.
@inline copyable_dimension(ℓu, ℓv, iu, iv, Nu, Nv) =
    equivalent_dimension(ℓu, ℓv, iu, iv, Nu, Nv) ||
    degenerate_dimension(ℓu, ℓv, Nu, Nv) ||
    expandable_source_dimension(ℓv, Nu)

# `u` and `v` place their nodes identically: same location, same extent, and windows
# selecting the same absolute indices.
@inline equivalent_dimension(ℓu, ℓv, iu, iv, Nu, Nv) =
    ℓu == ℓv && Nu == Nv && equivalent_index(iu, iv, Nu)

# Both fields span a single point and at least one carries no node there. A `Nothing`
# location has no node to interpolate to, so the single slab copies directly whatever the
# other field's location and absolute index are (e.g. a reduced field set from a windowed
# single-layer field, whose locations are `Nothing` vs `Center` and indices `:` vs `k:k`).
@inline degenerate_dimension(ℓu, ℓv, Nu, Nv) =
    Nu == Nv == 1 && (ℓu === Nothing || ℓv === Nothing)

# `v` carries no node, so its single value is replicated across `u`'s cells --- a 1D
# reference column set into a 3D field, say. Only the source may stretch, mirroring
# broadcasting, so a reduced `u` fed by a many-celled `v` is not copyable.
@inline expandable_source_dimension(ℓv, Nu) = ℓv === Nothing && Nu > 1

@inline equivalent_index(a, b, N) = a == b
@inline equivalent_index(::Colon, r::AbstractUnitRange, N) = first(r) == 1 && last(r) == N
@inline equivalent_index(r::AbstractUnitRange, ::Colon, N) = first(r) == 1 && last(r) == N

function copy_to_field!(u, v)
    # We implement some niceities in here that attempt to copy halo data,
    # and revert to copying just interior points if that fails.

    if size(u) != size(v)
        # `v` is reduced along at least one dimension (see `expandable_source_dimension`)
        # and stretches across `u`. Only interior points participate: `v` spans a single
        # point along a stretched dimension, so it has no halo data to lend there.
        if child_architecture(u) === child_architecture(v)
            interior(u) .= interior(v)
        else
            v_data = on_architecture(child_architecture(u), v.data)
            interior(u) .= interior(v_data, location(v), v.grid, v.indices)
        end

        return u
    end

    if child_architecture(u) === child_architecture(v)
        # Note: we could try to copy first halo point even when halo
        # regions are a different size. That's a bit more complicated than
        # the below so we leave it for the future.

        try # to copy halo regions along with interior data
            parent(u) .= parent(v)
        catch # this could fail if the halo regions are different sizes?
            # copy just the interior data
            interior(u) .= interior(v)
        end
    else
        v_data = on_architecture(child_architecture(u), v.data)

        # As above, we permit ourselves a little ambition and try to copy halo data:
        try
            parent(u) .= parent(v_data)
        catch
            interior(u) .= interior(v_data, location(v), v.grid, v.indices)
        end
    end

    return u
end

Base.copyto!(f::Field, src::Base.Broadcast.Broadcasted) = copyto!(interior(f), src)
Base.copyto!(f::Field, src::AbstractArray) = copyto!(interior(f), src)
Base.copyto!(f::Field, src::OffsetArray) = copyto!(interior(f), parent(src))
Base.copyto!(f::Field, src::Field) = copyto!(parent(f), parent(src))

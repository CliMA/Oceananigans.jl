using KernelAbstractions: @kernel, @index

using Oceananigans.Grids: node_names
using Oceananigans.Architectures: cpu_architecture, GPU, CPU, ReactantState

#####
##### Utilities
#####

function tuple_string(tup::Tuple)
    return join(string.(tup), ", ")
end

tuple_string(tup::Tuple{}) = ""

#####
##### set!
#####

set!(obj::AbstractField, ::Nothing) = nothing

function set!(Φ::NamedTuple; kwargs...)
    if hasproperty(Φ, :u) && hasproperty(Φ, :v)
        source_u = :u in keys(kwargs) ? kwargs[:u] : ZeroField()
        source_v = :v in keys(kwargs) ? kwargs[:v] : ZeroField()

        if (:u in keys(kwargs) || :v in keys(kwargs)) &&
           requires_single_component_quadfolded_vector_field_set(Φ.u, Φ.v, source_u, source_v)
            set_single_component_quadfolded_vector_fields!(Φ.u, Φ.v, source_u, source_v)

            for (fldname, value) in kwargs
                fldname in (:u, :v) && continue
                ϕ = getproperty(Φ, fldname)
                set!(ϕ, value)
            end

            return nothing
        end

        if requires_partial_quadfolded_vector_field_set(Φ.u, Φ.v, source_u, source_v)
            msg = string("Interpolating OctaHEALPix vector fields one component at a time is unsupported.", '\n',
                         "Set paired (u, v) fields together instead.")
            throw(ArgumentError(msg))
        end
    end

    if hasproperty(Φ, :u) &&
       hasproperty(Φ, :v) &&
       :u in keys(kwargs) &&
       :v in keys(kwargs) &&
       requires_paired_quadfolded_vector_field_set(Φ.u, Φ.v, kwargs[:u], kwargs[:v])
        set_paired_quadfolded_vector_fields!(Φ.u, Φ.v, kwargs[:u], kwargs[:v])

        for (fldname, value) in kwargs
            fldname in (:u, :v) && continue
            ϕ = getproperty(Φ, fldname)
            set!(ϕ, value)
        end

        return nothing
    end

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
    if hasproperty(dst, :u) && hasproperty(dst, :v)
        source_u = :u in keys(src) ? src.u : ZeroField()
        source_v = :v in keys(src) ? src.v : ZeroField()

        if (:u in keys(src) || :v in keys(src)) &&
           requires_single_component_quadfolded_vector_field_set(dst.u, dst.v, source_u, source_v)
            set_single_component_quadfolded_vector_fields!(dst.u, dst.v, source_u, source_v)

            for name in keys(src)
                name in (:u, :v) && continue
                set!(dst[name], src[name])
            end

            return dst
        end

        if (:u in keys(src) || :v in keys(src)) &&
           requires_partial_quadfolded_vector_field_set(dst.u, dst.v, source_u, source_v)
            msg = string("Interpolating OctaHEALPix vector fields one component at a time is unsupported.", '\n',
                         "Set paired (u, v) fields together instead.")
            throw(ArgumentError(msg))
        end

        if hasproperty(src, :u) &&
           hasproperty(src, :v) &&
           requires_paired_quadfolded_vector_field_set(dst.u, dst.v, src.u, src.v)
            set_paired_quadfolded_vector_fields!(dst.u, dst.v, src.u, src.v)

            for name in keys(src)
                name in (:u, :v) && continue
                set!(dst[name], src[name])
            end

            return dst
        end
    end

    for name in keys(src)
        set!(dst[name], src[name])
    end
    return dst
end

# This interface helps us do things like set distributed fields
set!(u::Field, f::Function) = set_to_function!(u, f)
set!(u::Field, a::Union{Array, OffsetArray}) = set_to_array!(u, a)

"""
    set!(u::Field, v::Field)

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
        copyto!(interior(u), a)
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
    if matching_field_storage_layout(u, v)
        copy_to_field!(u, v)
    else
        if single_component_quadfolded_u_field(u) &&
           single_component_quadfolded_u_field(v)
            companion_v = quadfolded_companion_field(u)
            set_single_component_quadfolded_vector_fields!(u, companion_v, v, ZeroField())
            return u
        elseif single_component_quadfolded_v_field(u) &&
               single_component_quadfolded_v_field(v)
            companion_u = quadfolded_companion_field(u)
            set_single_component_quadfolded_vector_fields!(companion_u, u, ZeroField(), v)
            return u
        elseif uses_quadfolded_vector_boundary_conditions(u) ||
               uses_quadfolded_vector_boundary_conditions(v)
            msg = string("Interpolating an OctaHEALPix vector field from a single source field is unsupported.", '\n',
                         "Set or fill paired (u, v) fields together instead.")
            throw(ArgumentError(msg))
        end

        # Fill halos on v's native architecture so distributed dispatch (if any) is used;
        # on_architecture would strip Distributed{CPU} to CPU while keeping distributed
        # boundary conditions, mismatching fill_halo_regions! dispatch.
        fill_halo_regions!(v)
        v_on_u = on_architecture(child_architecture(u), v)
        interpolate!(u, v_on_u)
    end

    return u
end

@inline function requires_paired_quadfolded_vector_field_set(to_u, to_v, from_u, from_v)
    paired_quadfolded_vector_fields =
        to_u isa Field{Face, Center} &&
        to_v isa Field{Center, Face} &&
        from_u isa Field{Face, Center} &&
        from_v isa Field{Center, Face} &&
        location(to_u)[3] == location(to_v)[3] == location(from_u)[3] == location(from_v)[3] &&
        uses_quadfolded_vector_boundary_conditions(from_u) &&
        uses_quadfolded_vector_boundary_conditions(from_v)

    paired_quadfolded_vector_fields || return false

    same_storage_layout =
        matching_field_storage_layout(to_u, from_u) &&
        matching_field_storage_layout(to_v, from_v)

    return !same_storage_layout
end

@inline single_component_quadfolded_u_field(source) =
    source isa Field{Face, Center} &&
    uses_quadfolded_vector_boundary_conditions(source)

@inline single_component_quadfolded_v_field(source) =
    source isa Field{Center, Face} &&
    uses_quadfolded_vector_boundary_conditions(source)

@inline function requires_quadfolded_vector_field_interpolation(to_field, from_field)
    return from_field isa Field &&
           location(to_field)[3] == location(from_field)[3] &&
           uses_quadfolded_vector_boundary_conditions(from_field) &&
           !matching_field_storage_layout(to_field, from_field)
end

@inline function requires_single_component_quadfolded_vector_field_set(to_u, to_v, from_u, from_v)
    u_requires = requires_quadfolded_vector_field_interpolation(to_u, from_u)
    v_requires = requires_quadfolded_vector_field_interpolation(to_v, from_v)

    return (u_requires &&
            from_v isa Union{ZeroField, OneField, ConstantField} &&
            single_component_quadfolded_u_field(from_u)) ||
           (v_requires &&
            from_u isa Union{ZeroField, OneField, ConstantField} &&
            single_component_quadfolded_v_field(from_v))
end

@inline function requires_partial_quadfolded_vector_field_set(to_u, to_v, from_u, from_v)
    u_requires = requires_quadfolded_vector_field_interpolation(to_u, from_u)
    v_requires = requires_quadfolded_vector_field_interpolation(to_v, from_v)
    paired_requires = requires_paired_quadfolded_vector_field_set(to_u, to_v, from_u, from_v)

    return (u_requires || v_requires) && !paired_requires
end

function set_paired_quadfolded_vector_fields!(to_u::Field{Face, Center, LZ},
                                              to_v::Field{Center, Face, LZ},
                                              from_u::Field{Face, Center, LZ},
                                              from_v::Field{Center, Face, LZ}) where LZ
    fill_halo_regions!((from_u, from_v))

    from_u_on_to_arch = on_architecture(child_architecture(to_u), from_u)
    from_v_on_to_arch = on_architecture(child_architecture(to_v), from_v)

    interpolate!((to_u, to_v), (from_u_on_to_arch, from_v_on_to_arch))

    return nothing
end

function set_single_component_quadfolded_vector_fields!(to_u::Field{Face, Center, LZ},
                                                        to_v::Field{Center, Face, LZ},
                                                        from_u::Field{Face, Center, LZ},
                                                        companion_v_source::Union{ZeroField, OneField, ConstantField}) where LZ
    companion_v = quadfolded_companion_field(from_u)
    set!(companion_v, companion_v_source)
    set_paired_quadfolded_vector_fields!(to_u, to_v, from_u, companion_v)
    return nothing
end

function set_single_component_quadfolded_vector_fields!(to_u::Field{Face, Center, LZ},
                                                        to_v::Field{Center, Face, LZ},
                                                        companion_u_source::Union{ZeroField, OneField, ConstantField},
                                                        from_v::Field{Center, Face, LZ}) where LZ
    companion_u = quadfolded_companion_field(from_v)
    set!(companion_u, companion_u_source)
    set_paired_quadfolded_vector_fields!(to_u, to_v, companion_u, from_v)
    return nothing
end

@inline function quadfolded_companion_field(source::Field{Face, Center, LZ}) where LZ
    _, _, zloc = instantiated_location(source)

    return Field((Center(), Face(), zloc),
                 source.grid;
                 indices = indices(source),
                 boundary_conditions =
                     Oceananigans.BoundaryConditions.FieldBoundaryConditions(
                         source.grid,
                         (Center(), Face(), zloc);
                         west = source.boundary_conditions.south,
                         east = source.boundary_conditions.north,
                         bottom = source.boundary_conditions.bottom,
                         top = source.boundary_conditions.top,
                         immersed = source.boundary_conditions.immersed))
end

@inline function quadfolded_companion_field(source::Field{Center, Face, LZ}) where LZ
    _, _, zloc = instantiated_location(source)

    return Field((Face(), Center(), zloc),
                 source.grid;
                 indices = indices(source),
                 boundary_conditions =
                     Oceananigans.BoundaryConditions.FieldBoundaryConditions(
                         source.grid,
                         (Face(), Center(), zloc);
                         south = source.boundary_conditions.west,
                         north = source.boundary_conditions.east,
                         bottom = source.boundary_conditions.bottom,
                         top = source.boundary_conditions.top,
                         immersed = source.boundary_conditions.immersed))
end

function matching_field_discretization(u, v)
    return size(u) == size(v) &&
           location(u) == location(v) &&
           equivalent_indices(indices(u), indices(v), size(u))
end

@inline function matching_field_storage_axes(u, v)
    return axes(u.data, 1) == axes(v.data, 1) &&
           axes(u.data, 2) == axes(v.data, 2) &&
           axes(u.data, 3) == axes(v.data, 3)
end

@inline matching_field_storage_layout(u, v) =
    matching_field_discretization(u, v) &&
    matching_field_storage_axes(u, v)

@inline equivalent_indices(ui::Tuple, vi::Tuple, sz::Tuple) =
    equivalent_index(ui[1], vi[1], sz[1]) &&
    equivalent_index(ui[2], vi[2], sz[2]) &&
    equivalent_index(ui[3], vi[3], sz[3])

@inline equivalent_index(a, b, N) = a == b
@inline equivalent_index(::Colon, r::AbstractUnitRange, N) = first(r) == 1 && last(r) == N
@inline equivalent_index(r::AbstractUnitRange, ::Colon, N) = first(r) == 1 && last(r) == N

function copy_to_field!(u, v)
    # We implement some niceities in here that attempt to copy halo data,
    # and revert to copying just interior points if that fails.

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
Base.copyto!(f::Field, src::Field) = copyto!(parent(f), parent(src))

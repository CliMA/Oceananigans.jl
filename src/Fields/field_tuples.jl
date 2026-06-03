using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Utils: KernelParameters, launch!
using KernelAbstractions: @kernel, @index
using LinearAlgebra: LinearAlgebra

const FieldTuple = Tuple{Field, Vararg{Field}}
const NamedFieldTuple = NamedTuple{S, <:FieldTuple} where S

Base.similar(ft::NamedFieldTuple) = map(similar, ft)

LinearAlgebra.norm(ft::NamedFieldTuple) = sqrt(sum(LinearAlgebra.norm(field)^2 for field in ft))

LinearAlgebra.dot(ft1::NamedFieldTuple, ft2::NamedFieldTuple) =
    sum(LinearAlgebra.dot(a, b) for (a, b) in zip(values(ft1), values(ft2)))

#####
##### `fill_halo_regions!` for tuples of `Field`
#####

@inline octahealpix_vector_halo_boundary_conditions(u, v) =
    (u.boundary_conditions.south,
     u.boundary_conditions.north,
     v.boundary_conditions.west,
     v.boundary_conditions.east)

@inline has_quadfolded_vector_halo_boundary_conditions(::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC) = true

@inline has_quadfolded_vector_halo_boundary_conditions(::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC) = true

@inline has_quadfolded_vector_halo_boundary_conditions(args...) = false

@inline any_quadfolded_vector_halo_boundary_conditions(::Oceananigans.BoundaryConditions.QCovZBC, args...) = true
@inline any_quadfolded_vector_halo_boundary_conditions(::Oceananigans.BoundaryConditions.QConZBC, args...) = true
@inline any_quadfolded_vector_halo_boundary_conditions(_, args...) = any_quadfolded_vector_halo_boundary_conditions(args...)
@inline any_quadfolded_vector_halo_boundary_conditions() = false

@inline function equivalent_octahealpix_vector_grids(grid_u, grid_v)
    return typeof(grid_u) === typeof(grid_v) &&
           Oceananigans.Grids.architecture(grid_u) === Oceananigans.Grids.architecture(grid_v) &&
           grid_u.Nx == grid_v.Nx &&
           grid_u.Ny == grid_v.Ny &&
           grid_u.Nz == grid_v.Nz &&
           typeof(grid_u.connectivity) === typeof(grid_v.connectivity)
end

@inline function fill_octahealpix_uv_halos_required(u, v)
    compatible_grids = u.grid === v.grid || equivalent_octahealpix_vector_grids(u.grid, v.grid)
    compatible_vertical_data_axes = axes(u.data, 3) == axes(v.data, 3)
    horizontally_full = u.indices[1] == Colon() &&
                        u.indices[2] == Colon() &&
                        v.indices[1] == Colon() &&
                        v.indices[2] == Colon()

    return compatible_grids &&
           compatible_vertical_data_axes &&
           horizontally_full &&
           has_quadfolded_vector_halo_boundary_conditions(octahealpix_vector_halo_boundary_conditions(u, v)...) &&
           u.grid isa Oceananigans.Grids.SphericalShellGrid &&
           u.grid.connectivity isa Oceananigans.Grids.OctaHEALPixConnectivity
end

@inline function octahealpix_namedtuple_uv_halo_pairs(fields::NamedTuple)
    field_names = collect(keys(fields))
    used_pair = falses(length(field_names))
    uv_pairs = Tuple{Symbol, Symbol}[]

    for u_index in eachindex(field_names)
        used_pair[u_index] && continue

        u_name = field_names[u_index]
        u = getproperty(fields, u_name)

        u isa Field{Face, Center} || continue

        for v_index in (u_index + 1):length(field_names)
            used_pair[v_index] && continue

            v_name = field_names[v_index]
            v = getproperty(fields, v_name)

            if v isa Field{Center, Face} &&
               fill_octahealpix_uv_halos_required(u, v)
                push!(uv_pairs, (u_name, v_name))
                used_pair[u_index] = true
                used_pair[v_index] = true
                break
            end
        end
    end

    return Tuple(uv_pairs)
end

@inline function octahealpix_tuple_uv_halo_pairs(fields::Tuple)
    used_pair = falses(length(fields))
    uv_pairs = NTuple{2, Int}[]

    for u_index in eachindex(fields)
        used_pair[u_index] && continue

        u = fields[u_index]
        u isa Field{Face, Center} || continue

        for v_index in (u_index + 1):length(fields)
            used_pair[v_index] && continue

            v = fields[v_index]

            if v isa Field{Center, Face} &&
               fill_octahealpix_uv_halos_required(u, v)
                push!(uv_pairs, (u_index, v_index))
                used_pair[u_index] = true
                used_pair[v_index] = true
                break
            end
        end
    end

    return Tuple(uv_pairs)
end

@inline octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, ::Val{:covariant}) =
    Oceananigans.Grids.octahealpix_covariant_xface_halo_source(i, j, Nx, Ny, connectivity)

@inline octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, ::Val{:covariant}) =
    Oceananigans.Grids.octahealpix_covariant_yface_halo_source(i, j, Nx, Ny, connectivity)

@inline octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, ::Val{:contravariant}) =
    Oceananigans.Grids.octahealpix_contravariant_xface_halo_source(i, j, Nx, Ny, connectivity)

@inline octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, ::Val{:contravariant}) =
    Oceananigans.Grids.octahealpix_contravariant_yface_halo_source(i, j, Nx, Ny, connectivity)

@kernel function _fill_octahealpix_u_vector_halos!(u, v, connectivity, Nx, Ny, transform, halo_sign)
    i, j, k = @index(Global, NTuple)

    inside_i = (i >= 1) & (i <= Nx)
    inside_j = (j >= 1) & (j <= Ny)
    interior_point = inside_i & inside_j

    source_kind, source_i, source_j, sign =
        Oceananigans.Grids.octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, transform)

    @inbounds halo_u = ifelse(source_kind == 1,
                              sign * u[source_i, source_j, k],
                              sign * v[source_i, source_j, k])
    @inbounds halo_u *= halo_sign
    @inbounds u[i, j, k] = ifelse(interior_point, u[i, j, k], halo_u)
end

@kernel function _fill_octahealpix_v_vector_halos!(u, v, connectivity, Nx, Ny, transform, halo_sign)
    i, j, k = @index(Global, NTuple)

    inside_i = (i >= 1) & (i <= Nx)
    inside_j = (j >= 1) & (j <= Ny)
    interior_point = inside_i & inside_j

    source_kind, source_i, source_j, sign =
        Oceananigans.Grids.octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, transform)

    @inbounds halo_v = ifelse(source_kind == 1,
                              sign * u[source_i, source_j, k],
                              sign * v[source_i, source_j, k])
    @inbounds halo_v *= halo_sign
    @inbounds v[i, j, k] = ifelse(interior_point, v[i, j, k], halo_v)
end

function fill_octahealpix_vector_halos!(u::Field{Face, Center, Center},
                                        v::Field{Center, Face, Center},
                                        transform,
                                        halo_sign)
    grid = u.grid
    params = KernelParameters(axes(u.data, 1),
                              axes(u.data, 2),
                              axes(u.data, 3))

    launch!(Oceananigans.Grids.architecture(grid), grid, params,
            _fill_octahealpix_u_vector_halos!, u.data, v.data, grid.connectivity, grid.Nx, grid.Ny, transform, halo_sign)

    params = KernelParameters(axes(v.data, 1),
                              axes(v.data, 2),
                              axes(v.data, 3))

    launch!(Oceananigans.Grids.architecture(grid), grid, params,
            _fill_octahealpix_v_vector_halos!, u.data, v.data, grid.connectivity, grid.Nx, grid.Ny, transform, halo_sign)

    return nothing
end

function fill_octahealpix_vector_halos!(u::Field{Face, Center, LZ},
                                        v::Field{Center, Face, LZ},
                                        transform,
                                        halo_sign) where LZ
    grid = u.grid
    params = KernelParameters(axes(u.data, 1),
                              axes(u.data, 2),
                              axes(u.data, 3))

    launch!(Oceananigans.Grids.architecture(grid), grid, params,
            _fill_octahealpix_u_vector_halos!, u.data, v.data, grid.connectivity, grid.Nx, grid.Ny, transform, halo_sign)

    params = KernelParameters(axes(v.data, 1),
                              axes(v.data, 2),
                              axes(v.data, 3))

    launch!(Oceananigans.Grids.architecture(grid), grid, params,
            _fill_octahealpix_v_vector_halos!, u.data, v.data, grid.connectivity, grid.Nx, grid.Ny, transform, halo_sign)

    return nothing
end

fill_octahealpix_uv_halos!(u::Field{Face, Center, Center}, v::Field{Center, Face, Center}, halo_sign) =
    fill_octahealpix_vector_halos!(u, v, Val(:covariant), halo_sign)

fill_octahealpix_contravariant_vector_halos!(u::Field{Face, Center, Center}, v::Field{Center, Face, Center}, halo_sign) =
    fill_octahealpix_vector_halos!(u, v, Val(:contravariant), halo_sign)

fill_octahealpix_uv_halos!(u::Field{Face, Center, LZ}, v::Field{Center, Face, LZ}, halo_sign) where LZ =
    fill_octahealpix_vector_halos!(u, v, Val(:covariant), halo_sign)

fill_octahealpix_contravariant_vector_halos!(u::Field{Face, Center, LZ}, v::Field{Center, Face, LZ}, halo_sign) where LZ =
    fill_octahealpix_vector_halos!(u, v, Val(:contravariant), halo_sign)

function fill_octahealpix_covariant_vector_halo_regions!(u::Field{Face, Center, Center},
                                                         v::Field{Center, Face, Center},
                                                         halo_sign,
                                                         args...; kwargs...)
    fill_vertical_halos_only!(u, args...; kwargs...)
    fill_vertical_halos_only!(v, args...; kwargs...)
    fill_octahealpix_uv_halos!(u, v, halo_sign)
    return nothing
end

function fill_octahealpix_covariant_vector_halo_regions!(u::Field{Face, Center, LZ},
                                                         v::Field{Center, Face, LZ},
                                                         halo_sign,
                                                         args...; kwargs...) where LZ
    fill_vertical_halos_only!(u, args...; kwargs...)
    fill_vertical_halos_only!(v, args...; kwargs...)
    fill_octahealpix_uv_halos!(u, v, halo_sign)
    return nothing
end

function fill_octahealpix_contravariant_vector_halo_regions!(u::Field{Face, Center, Center},
                                                             v::Field{Center, Face, Center},
                                                             halo_sign,
                                                             args...; kwargs...)
    fill_vertical_halos_only!(u, args...; kwargs...)
    fill_vertical_halos_only!(v, args...; kwargs...)
    fill_octahealpix_contravariant_vector_halos!(u, v, halo_sign)
    return nothing
end

function fill_octahealpix_contravariant_vector_halo_regions!(u::Field{Face, Center, LZ},
                                                             v::Field{Center, Face, LZ},
                                                             halo_sign,
                                                             args...; kwargs...) where LZ
    fill_vertical_halos_only!(u, args...; kwargs...)
    fill_vertical_halos_only!(v, args...; kwargs...)
    fill_octahealpix_contravariant_vector_halos!(u, v, halo_sign)
    return nothing
end

function quadfolded_vector_halo_sign(south_bc, north_bc, west_bc, east_bc)
    south_sign = south_bc.condition
    north_sign = north_bc.condition
    west_sign = west_bc.condition
    east_sign = east_bc.condition

    if south_sign == north_sign == west_sign == east_sign
        return south_sign
    end

    msg = string("Inconsistent QuadFolded vector halo sign payloads are unsupported.", '\n',
                 "Received seam BC signs: south = ", summary(south_bc), ", north = ", summary(north_bc),
                 ", west = ", summary(west_bc), ", east = ", summary(east_bc), ".")
    throw(ArgumentError(msg))
end

@inline function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, Center},
                                                       v::Field{Center, Face, Center},
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       south_bc,
                                                       north_bc,
                                                       west_bc,
                                                       east_bc,
                                                       args...; kwargs...)
    halo_sign = quadfolded_vector_halo_sign(south_bc, north_bc, west_bc, east_bc)
    fill_octahealpix_covariant_vector_halo_regions!(u, v, halo_sign, args...; kwargs...)
    return nothing
end

@inline function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, LZ},
                                                       v::Field{Center, Face, LZ},
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       ::Oceananigans.BoundaryConditions.QCovZBC,
                                                       south_bc,
                                                       north_bc,
                                                       west_bc,
                                                       east_bc,
                                                       args...; kwargs...) where LZ
    halo_sign = quadfolded_vector_halo_sign(south_bc, north_bc, west_bc, east_bc)
    fill_octahealpix_covariant_vector_halo_regions!(u, v, halo_sign, args...; kwargs...)
    return nothing
end

@inline function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, Center},
                                                       v::Field{Center, Face, Center},
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       south_bc,
                                                       north_bc,
                                                       west_bc,
                                                       east_bc,
                                                       args...; kwargs...)
    halo_sign = quadfolded_vector_halo_sign(south_bc, north_bc, west_bc, east_bc)
    fill_octahealpix_contravariant_vector_halo_regions!(u, v, halo_sign, args...; kwargs...)
    return nothing
end

@inline function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, LZ},
                                                       v::Field{Center, Face, LZ},
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       ::Oceananigans.BoundaryConditions.QConZBC,
                                                       south_bc,
                                                       north_bc,
                                                       west_bc,
                                                       east_bc,
                                                       args...; kwargs...) where LZ
    halo_sign = quadfolded_vector_halo_sign(south_bc, north_bc, west_bc, east_bc)
    fill_octahealpix_contravariant_vector_halo_regions!(u, v, halo_sign, args...; kwargs...)
    return nothing
end

function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, Center},
                                               v::Field{Center, Face, Center},
                                               south_bc,
                                               north_bc,
                                               west_bc,
                                               east_bc,
                                               args...; kwargs...)
    msg = string("Unsupported mixed QuadFolded vector boundary conditions for OctaHEALPix halo fill.", '\n',
                 "Received seam BCs: south = ", summary(south_bc), ", north = ", summary(north_bc),
                 ", west = ", summary(west_bc), ", east = ", summary(east_bc), ".")
    throw(ArgumentError(msg))
end

function fill_octahealpix_vector_halo_regions!(u::Field{Face, Center, LZ},
                                               v::Field{Center, Face, LZ},
                                               south_bc,
                                               north_bc,
                                               west_bc,
                                               east_bc,
                                               args...; kwargs...) where LZ
    msg = string("Unsupported mixed QuadFolded vector boundary conditions for OctaHEALPix halo fill.", '\n',
                 "Received seam BCs: south = ", summary(south_bc), ", north = ", summary(north_bc),
                 ", west = ", summary(west_bc), ", east = ", summary(east_bc), ".")
    throw(ArgumentError(msg))
end

@inline flattened_unique_values(::Tuple{}) = tuple()

"""
    flattened_unique_values(a::NamedTuple)

Return values of the (possibly nested) `NamedTuple` `a`,
flattened into a single tuple, with duplicate entries removed.
"""
@inline function flattened_unique_values(a::Union{NamedTuple, Tuple})
    tupled = Tuple(tuplify(ai) for ai in a)
    flattened = flatten_tuple(tupled)

    # Alternative implementation of `unique` for tuples that uses === comparison, rather than ==
    seen = []
    return Tuple(last(push!(seen, f)) for f in flattened if !any(f === s for s in seen))
end

const FullField = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple{<:Colon, <:Colon, <:Colon}}

# Utility for extracting values from nested NamedTuples
@inline tuplify(a::NamedTuple) = Tuple(tuplify(ai) for ai in a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()

"""
    fill_halo_regions!(fields::NamedTuple, args...; kwargs...)

Fill halo regions for all `fields`. The algorithm:

  1. Flattens fields, extracting `values` if the field is `NamedTuple`, and removing
     duplicate entries to avoid "repeated" halo filling.

  2. Filters fields into three categories:
     i. ReducedFields with non-trivial boundary conditions;
     ii. Fields with non-trivial indices and boundary conditions;
     iii. Fields spanning the whole grid with non-trivial boundary conditions.

  3. Halo regions for every `ReducedField` and windowed fields are filled independently.

  4. In every direction, the halo regions in each of the remaining `Field` tuple
     are filled simultaneously.
"""
function BoundaryConditions.fill_halo_regions!(fields::Union{NamedTuple, Tuple}, args...; kwargs...)
    if fields isa NamedTuple
        uv_pairs = octahealpix_namedtuple_uv_halo_pairs(fields)

        if !isempty(uv_pairs)
            for (u_name, v_name) in uv_pairs
                fill_halo_regions!((getproperty(fields, u_name), getproperty(fields, v_name)), args...; kwargs...)
            end

            for name in keys(fields)
                pair_filled = any(name === u_name || name === v_name for (u_name, v_name) in uv_pairs)
                pair_filled && continue

                @inbounds fill_halo_regions!(getproperty(fields, name), args...; kwargs...)
            end

            return nothing
        end
    elseif fields isa Tuple && length(fields) >= 2
        uv_pairs = octahealpix_tuple_uv_halo_pairs(fields)

        if !isempty(uv_pairs)
            for (u_index, v_index) in uv_pairs
                fill_halo_regions!((fields[u_index], fields[v_index]), args...; kwargs...)
            end

            for i in eachindex(fields)
                pair_filled = any(i == u_index || i == v_index for (u_index, v_index) in uv_pairs)
                pair_filled && continue

                @inbounds fill_halo_regions!(fields[i], args...; kwargs...)
            end

            return nothing
        end
    end

    for i in eachindex(fields)
        @inbounds fill_halo_regions!(fields[i], args...; kwargs...)
    end

    return nothing
end

function BoundaryConditions.fill_halo_regions!(fields::Tuple{<:Field{Face, Center, LZ}, <:Field{Center, Face, LZ}}, args...; kwargs...) where LZ
    u, v = fields
    vector_halo_bcs = octahealpix_vector_halo_boundary_conditions(u, v)

    if any_quadfolded_vector_halo_boundary_conditions(vector_halo_bcs...) &&
       !has_quadfolded_vector_halo_boundary_conditions(vector_halo_bcs...)
        fill_octahealpix_vector_halo_regions!(u, v, vector_halo_bcs..., args...; kwargs...)
    end

    if fill_octahealpix_uv_halos_required(u, v)
        fill_octahealpix_vector_halo_regions!(u, v,
                                              vector_halo_bcs...,
                                              vector_halo_bcs...,
                                              args...; kwargs...)
        return nothing
    end

    fill_halo_regions!(u, args...; kwargs...)
    fill_halo_regions!(v, args...; kwargs...)

    return nothing
end

#####
##### Tracer names
#####

# TODO: This code belongs in the Models module

"Returns true if the first three elements of `names` are `(:u, :v, :w)`."
has_velocities(names) = :u == names[1] && :v == names[2] && :w == names[3]

# Tuples of length 0-2 cannot contain velocity fields
has_velocities(::Tuple{}) = false
has_velocities(::Tuple{X}) where X = false
has_velocities(::Tuple{X, Y}) where {X, Y} = false

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = has_velocities(names) ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

#####
##### Validation
#####

validate_field_grid(grid, field) = grid === field.grid

validate_field_grid(grid, field_tuple::NamedTuple) =
    all(validate_field_grid(grid, field) for field in field_tuple)

"""
    validate_field_tuple_grid(tuple_name, field_tuple, grid)

Validates the grids associated with grids in the (possibly nested) `field_tuple`,
and returns `field_tuple` if validation succeeds.
"""
function validate_field_tuple_grid(tuple_name, field_tuple, grid)

    all(validate_field_grid(grid, field) for field in field_tuple) ||
        throw(ArgumentError("Model grid and $tuple_name grid are not identical! " *
                            "Check that the grid used to construct $tuple_name has the correct halo size."))

    return nothing
end

#####
##### Velocity fields tuples
#####

"""
    VelocityFields(grid, user_bcs = NamedTuple())

Return a `NamedTuple` with fields `u`, `v`, `w` initialized on `grid`.
Boundary conditions `bcs` may be specified via a named tuple of
`FieldBoundaryCondition`s.
"""
function VelocityFields(grid::AbstractGrid, user_bcs = NamedTuple())

    template = FieldBoundaryConditions()

    default_bcs = (
        u = regularize_field_boundary_conditions(template, grid, :u),
        v = regularize_field_boundary_conditions(template, grid, :v),
        w = regularize_field_boundary_conditions(template, grid, :w)
    )

    bcs = merge(default_bcs, user_bcs)

    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid, boundary_conditions=bcs.w)

    return (u=u, v=v, w=w)
end

#####
##### Tracer fields tuples
#####

"""
    TracerFields(tracer_names, grid, user_bcs)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Boundary conditions `user_bcs`
may be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(tracer_names, grid, user_bcs)
    default_bcs = NamedTuple(name => FieldBoundaryConditions(grid, (Center(), Center(), Center())) for name in tracer_names)
    bcs = merge(default_bcs, user_bcs) # provided bcs overwrite defaults
    return NamedTuple(c => CenterField(grid, boundary_conditions=bcs[c]) for c in tracer_names)
end

"""
    TracerFields(tracer_names, grid; kwargs...)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Fields may be passed via optional keyword arguments `kwargs`
for each field.
"""
TracerFields(tracer_names, grid; kwargs...) =
    NamedTuple(c => c ∈ keys(kwargs) ? kwargs[c] : CenterField(grid) for c in tracer_names)

# 'Nothing', or empty tracer fields
TracerFields(::Union{Tuple{}, Nothing}, grid, bcs) = NamedTuple()

"Shortcut constructor for empty tracer fields."
TracerFields(::NamedTuple{(), Tuple{}}, grid, bcs) = NamedTuple()

#####
##### Helper functions for NonhydrostaticModel constructor
#####

VelocityFields(::Nothing, grid, bcs) = VelocityFields(grid, bcs)

"""
    VelocityFields(proposed_velocities::NamedTuple{(:u, :v, :w)}, grid, bcs)

Return a `NamedTuple` of velocity fields, overwriting boundary conditions
in `proposed_velocities` with corresponding fields in the `NamedTuple` `bcs`.
"""
function VelocityFields(proposed_velocities::NamedTuple{(:u, :v, :w)}, grid, bcs)

    validate_field_tuple_grid("velocities", proposed_velocities, grid)

    u = XFaceField(grid, boundary_conditions=bcs.u, data=proposed_velocities.u.data)
    v = YFaceField(grid, boundary_conditions=bcs.v, data=proposed_velocities.v.data)
    w = ZFaceField(grid, boundary_conditions=bcs.w, data=proposed_velocities.w.data)

    return (u=u, v=v, w=w)
end

"""
    TracerFields(proposed_tracers::NamedTuple, grid, bcs)

Return a `NamedTuple` of tracers, overwriting boundary conditions
in `proposed_tracers` with corresponding fields in the `NamedTuple` `bcs`.
"""
function TracerFields(proposed_tracers::NamedTuple, grid, bcs)

    validate_field_tuple_grid("tracers", proposed_tracers, grid)

    tracer_names = propertynames(proposed_tracers)
    tracer_fields = Tuple(CenterField(grid, boundary_conditions=bcs[c], data=proposed_tracers[c].data) for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

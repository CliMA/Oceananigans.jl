using Oceananigans.Architectures: CPU, GPU, architecture, convert_to_device
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, NoKernelFBC, regularize_field_boundary_conditions,
    tupled_fill_halo_regions!, needs_halo_fill_args
using Oceananigans.Utils: foreach_element
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

@inline flattened_unique_values(::Tuple{}) = tuple()

"""
$(TYPEDSIGNATURES)

Return values of the (possibly nested) `NamedTuple` `a`,
flattened into a single tuple, with duplicate entries removed.
"""
@inline function flattened_unique_values(a::Union{NamedTuple, Tuple})
    flattened = flatten_tuple(map(tuplify, values(a)))
    return collect_unique_values(flattened, ())
end

@inline collect_unique_values(::Tuple{}, accumulated::Tuple) = accumulated

@inline function collect_unique_values(remaining::Tuple, accumulated::Tuple)
    head = first(remaining)
    rest = Base.tail(remaining)
    accumulated = is_contained(head, accumulated) ? accumulated : (accumulated..., head)
    return collect_unique_values(rest, accumulated)
end

@inline is_contained(value, ::Tuple{}) = false
# `is_contained` verifies if an object is duplicated in the tuple, thus needing the equality `===` operator.
@inline is_contained(value, accumulated::Tuple) = value === first(accumulated) || is_contained(value, Base.tail(accumulated))

const FullField = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple{<:Colon, <:Colon, <:Colon}}

# Utility for extracting values from nested NamedTuples and Tuples.
@inline tuplify(a::NamedTuple) = map(tuplify, values(a))
@inline tuplify(a::Tuple)      = map(tuplify, a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()

"""
$(TYPEDSIGNATURES)

Fill the halo regions of all `fields`. Fields that share a grid, location and halo-filling kernels,
and whose boundary conditions do not read `args`, are filled together with one kernel launch per side.
The remaining fields are filled one by one.
"""
@inline BoundaryConditions.fill_halo_regions!(fields::NamedTuple, args::Vararg{Any, N}; kwargs...) where N =
    fill_halo_regions!(values(fields), args...; kwargs...)

@inline BoundaryConditions.fill_halo_regions!(::Tuple{}, args...; kwargs...) = nothing

@inline function BoundaryConditions.fill_halo_regions!(fields::Tuple, args::Vararg{Any, N}; kwargs...) where N
    groups = halo_fill_groups(select_fields(tupled_fillable, fields))
    individual_fields = select_fields(!tupled_fillable, fields)
    foreach_element(fill_halo_group!, groups, args...; kwargs...)
    foreach_element(fill_halo_regions!, individual_fields, args...; kwargs...)
    return nothing
end

#####
##### Tupled halo filling
#####

const CenterOrFace = Union{Center, Face}
const SerialGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:Union{CPU, GPU}}
const TupledFillField = Field{<:CenterOrFace, <:CenterOrFace, <:CenterOrFace, <:Any, <:SerialGrid, <:Tuple{Colon, Colon, Colon}, <:AbstractOffsetArray}

# Whether the halos of `field` may be filled together with those of other fields
@inline tupled_fillable(field) = false
@inline tupled_fillable(field::TupledFillField) = tupled_fillable(field.boundary_conditions)
@inline tupled_fillable(bcs::FieldBoundaryConditions) = !needs_halo_fill_args(bcs)
@inline tupled_fillable(::NoKernelFBC) = false

# Whether two fields launch the same halo-filling kernels
@inline shares_halo_fill(a, b) = false

@inline shares_halo_fill(a::Field{LX, LY, LZ, <:Any, G}, b::Field{LX, LY, LZ, <:Any, G}) where {LX, LY, LZ, G} =
    typeof(a.boundary_conditions.kernels) === typeof(b.boundary_conditions.kernels)

# Type-stable `filter` for tuples, whose `predicate` depends only on the types of the elements
@inline select_fields(predicate, ::Tuple{}) = ()

@inline function select_fields(predicate::P, fields::Tuple) where P
    head = first(fields)
    selected = select_fields(predicate, Base.tail(fields))
    return predicate(head) ? (head, selected...) : selected
end

@inline halo_fill_groups(::Tuple{}) = ()

@inline function halo_fill_groups(fields::Tuple)
    head = first(fields)
    group = select_fields(field -> shares_halo_fill(head, field), fields)
    others = select_fields(field -> !shares_halo_fill(head, field), fields)
    return (group, halo_fill_groups(others)...)
end

@inline fill_halo_group!(group::Tuple{Any}, args::Vararg{Any, N}; kwargs...) where N =
    fill_halo_regions!(first(group), args...; kwargs...)

function fill_halo_group!(group::Tuple, args::Vararg{Any, N}; kwargs...) where N
    field = first(group)
    grid = field.grid

    all(f -> f.grid === grid, group) || return foreach_element(fill_halo_regions!, group, args...; kwargs...)

    data = map(f -> f.data, group)
    bcs = map(f -> f.boundary_conditions.ordered_bcs, group)

    GC.@preserve group begin
        device_data, device_bcs, device_grid = convert_to_device(architecture(grid), (data, bcs, grid))
        tupled_fill_halo_regions!(device_data, field.boundary_conditions.kernels, device_bcs,
                                  instantiated_location(field), device_grid; kwargs...)
    end

    return nothing
end

#####
##### Tracer names
#####

# TODO: This code belongs in the Models module

"Returns true if the first three elements of `names` are `(:u, :v, :w)`."
@inline has_velocities(names) = :u == names[1] && :v == names[2] && :w == names[3]

# Tuples of length 0-2 cannot contain velocity fields
has_velocities(::Tuple{}) = false
has_velocities(::Tuple{X}) where X = false
has_velocities(::Tuple{X, Y}) where {X, Y} = false

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
@inline tracernames(names::NTuple{N, Symbol}) where N = has_velocities(names) ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

#####
##### Validation
#####

validate_field_grid(grid, field) = grid === field.grid

validate_field_grid(grid, field_tuple::NamedTuple) =
    all(validate_field_grid(grid, field) for field in field_tuple)

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Boundary conditions `user_bcs`
may be specified via a named tuple of `FieldBoundaryCondition`s.
"""
Base.@constprop :aggressive function TracerFields(tracer_names, grid, user_bcs)
    default_bcs = named_tuple(name -> FieldBoundaryConditions(grid, (Center(), Center(), Center())), tracer_names)
    bcs = merge(default_bcs, user_bcs) # provided bcs overwrite defaults

    return named_tuple(tracer_names) do c
        Base.@constprop :aggressive
        CenterField(grid, boundary_conditions=bcs[c])
    end
end

"""
$(TYPEDSIGNATURES)

Return a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on `grid`. Fields may be passed via optional keyword arguments `kwargs`
for each field.
"""
Base.@constprop :aggressive function TracerFields(tracer_names, grid; kwargs...)
    return named_tuple(tracer_names) do c
        Base.@constprop :aggressive
        c ∈ keys(kwargs) ? kwargs[c] : CenterField(grid)
    end
end

# 'Nothing', or empty tracer fields
TracerFields(::Union{Tuple{}, Nothing}, grid, bcs) = NamedTuple()

"Shortcut constructor for empty tracer fields."
TracerFields(::NamedTuple{(), Tuple{}}, grid, bcs) = NamedTuple()

#####
##### Helper functions for NonhydrostaticModel constructor
#####

VelocityFields(::Nothing, grid, bcs) = VelocityFields(grid, bcs)

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Return a `NamedTuple` of tracers, overwriting boundary conditions
in `proposed_tracers` with corresponding fields in the `NamedTuple` `bcs`.
"""
function TracerFields(proposed_tracers::NamedTuple, grid, bcs)

    validate_field_tuple_grid("tracers", proposed_tracers, grid)

    tracer_names = propertynames(proposed_tracers)

    return named_tuple(tracer_names) do c
        Base.@constprop :aggressive
        CenterField(grid, boundary_conditions=bcs[c], data=proposed_tracers[c].data)
    end
end

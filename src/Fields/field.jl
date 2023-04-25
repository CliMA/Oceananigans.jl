using Oceananigans.BoundaryConditions: OBC, MCBC
using Oceananigans.Grids: parent_index_range, index_range_offset, default_indices, all_indices, validate_indices

using Adapt
using KernelAbstractions: @kernel, @index
using Base: @propagate_inbounds

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Statistics: norm, mean, mean!
import Base: ==

#####
##### The bees knees
#####

struct Field{LX, LY, LZ, O, G, I, D, T, B, S, F} <: AbstractField{LX, LY, LZ, G, T, 3}
    grid :: G
    data :: D
    boundary_conditions :: B
    indices :: I
    operand :: O
    status :: S
    boundary_buffers :: F

    # Inner constructor that does not validate _anything_!
    function Field{LX, LY, LZ}(grid::G, data::D, bcs::B, indices::I, op::O, status::S, buffers::F) where {LX, LY, LZ, G, D, B, O, S, I, F}
        T = eltype(data)
        return new{LX, LY, LZ, O, G, I, D, T, B, S, F}(grid, data, bcs, indices, op, status, buffers)
    end
end

#####
##### Constructor utilities
#####

function validate_field_data(loc, data, grid, indices)
    Fx, Fy, Fz = total_size(grid, loc, indices)

    if size(data) != (Fx, Fy, Fz)
        LX, LY, LZ = loc    
        e = "Cannot construct field at ($LX, $LY, $LZ) with size(data)=$(size(data)). " *
            "`data` must have size ($Fx, $Fy, $Fz)."
        throw(ArgumentError(e))
    end

    return nothing
end

validate_boundary_condition_location(bc, ::Center, side) = nothing                         # anything goes for centers
validate_boundary_condition_location(::Union{OBC, Nothing, MCBC}, ::Face, side) = nothing  # only open, connected or nothing on faces
validate_boundary_condition_location(::Nothing, ::Nothing, side) = nothing                 # its nothing or nothing
validate_boundary_condition_location(bc, loc, side) = # everything else is wrong!
    throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc)!"))

validate_boundary_conditions(loc, grid, ::Missing) = nothing
validate_boundary_conditions(loc, grid, ::Nothing) = nothing

function validate_boundary_conditions(loc, grid, bcs)
    sides = (:east, :west, :north, :south, :bottom, :top)
    directions = (1, 1, 2, 2, 3, 3)

    for (side, dir) in zip(sides, directions)
        topo = topology(grid, dir)()
        ℓ = loc[dir]()
        bc = getproperty(bcs, side)

        # Check that boundary condition jives with the grid topology
        validate_boundary_condition_topology(bc, topo, side)

        # Check that boundary condition is valid given field location
        topo isa Bounded && validate_boundary_condition_location(bc, ℓ, side)

        # Check that boundary condition arrays, if used, are on the right architecture
        validate_boundary_condition_architecture(bc, architecture(grid), side)
    end

    return nothing
end

#####
##### Some basic constructors
#####

# Common outer constructor for all field flavors that performs input validation
function Field(loc::Tuple, grid::AbstractGrid, data, bcs, indices, op=nothing, status=nothing)
    @apply_regionally indices = validate_indices(indices, loc, grid)
    @apply_regionally validate_field_data(loc, data, grid, indices)
    @apply_regionally validate_boundary_conditions(loc, grid, bcs)
    buffers = FieldBoundaryBuffers(grid, data, bcs)
    LX, LY, LZ = loc
    return Field{LX, LY, LZ}(grid, data, bcs, indices, op, status, buffers)
end

"""
    Field{LX, LY, LZ}(grid::AbstractGrid,
                      T::DataType=eltype(grid); kw...) where {LX, LY, LZ}

Construct a `Field` on `grid` with data type `T` at the location `(LX, LY, LZ)`.
Each of `(LX, LY, LZ)` is either `Center` or `Face` and determines the field's
location in `(x, y, z)` respectively.

Keyword arguments
=================

- `data :: OffsetArray`: An offset array with the fields data. If nothing is provided the
  field is filled with zeros.
  - `boundary_conditions`: If nothing is provided, then field is created using the default
  boundary conditions via [`FieldBoundaryConditions`](@ref).
- `indices`: Used to prescribe where a reduced field lives on. For example, at which `k` index
  does a two-dimensional ``x``-``y`` field lives on. Default: `(:, :, :)`.

Example
=======

A field at location `(Face, Face, Center)`.

```jldoctest fields
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 3, 4), extent=(1, 1, 1));

julia> ω = Field{Face, Face, Center}(grid)
2×3×4 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 8×9×10 OffsetArray(::Array{Float64, 3}, -2:5, -2:6, -2:7) with eltype Float64 with indices -2:5×-2:6×-2:7
    └── max=0.0, min=0.0, mean=0.0
```

Now, using `indices` we can create a two dimensional ``x``-``y`` field at location
`(Face, Face, Center)` to compute, e.g., the vertical vorticity ``∂v/∂x - ∂u/∂y``
at the fluid's surface ``z = 0``, which for `Center` corresponds to `k = Nz`.

```jldoctest fields
julia> u = XFaceField(grid); v = YFaceField(grid);

julia> ωₛ = Field(∂x(v) - ∂y(u), indices=(:, :, grid.Nz))
2×3×1 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: ZeroFlux
├── operand: BinaryOperation at (Face, Face, Center)
├── status: time=0.0
└── data: 8×9×1 OffsetArray(::Array{Float64, 3}, -2:5, -2:6, 4:4) with eltype Float64 with indices -2:5×-2:6×4:4
    └── max=0.0, min=0.0, mean=0.0

julia> compute!(ωₛ)
2×3×1 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: ZeroFlux
├── operand: BinaryOperation at (Face, Face, Center)
├── status: time=0.0
└── data: 8×9×1 OffsetArray(::Array{Float64, 3}, -2:5, -2:6, 4:4) with eltype Float64 with indices -2:5×-2:6×4:4
    └── max=0.0, min=0.0, mean=0.0
```
"""
function Field{LX, LY, LZ}(grid::AbstractGrid,
                           T::DataType=eltype(grid);
                           kw...) where {LX, LY, LZ}

    return Field((LX, LY, LZ), grid, T; kw...)
end

function Field(loc::Tuple,
               grid::AbstractGrid,
               T::DataType = eltype(grid);
               indices = default_indices(3),
               data = new_data(T, grid, loc, validate_indices(indices, loc, grid)),
               boundary_conditions = FieldBoundaryConditions(grid, loc, validate_indices(indices, loc, grid)),
               operand = nothing,
               status = nothing)

    return Field(loc, grid, data, boundary_conditions, indices, operand, status)
end
    
Field(z::ZeroField; kw...) = z
Field(f::Field; indices=f.indices) = view(f, indices...) # hmm...

"""
    CenterField(grid; kw...)

Return a `Field{Center, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
CenterField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center, Center, Center), grid, T; kw...)

"""
    XFaceField(grid; kw...)

Return a `Field{Face, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
XFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Face, Center, Center), grid, T; kw...)

"""
    YFaceField(grid; kw...)

Return a `Field{Center, Face, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
YFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center, Face, Center), grid, T; kw...)

"""
    ZFaceField(grid; kw...)

Return a `Field{Center, Center, Face}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
ZFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center, Center, Face), grid, T; kw...)

#####
##### Field utils
#####

# Canonical `similar` for Field (doesn't transfer boundary conditions)
function Base.similar(f::Field, grid=f.grid)
    loc = location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(parent(f)), grid, loc, f.indices),
                 FieldBoundaryConditions(grid, loc, f.indices),
                 f.indices,
                 f.operand,
                 deepcopy(f.status))
end

"""
    offset_windowed_data(data, loc, grid, indices)

Return an `OffsetArray` of a `view` of `parent(data)` with `indices`.

If `indices === (:, :, :)`, return an `OffsetArray` of `parent(data)`.
"""
function offset_windowed_data(data, Loc, grid, indices)
    halo = halo_size(grid)
    topo = map(instantiate, topology(grid))
    loc = map(instantiate, Loc)

    if indices isa typeof(default_indices(3))
        windowed_parent = parent(data)
    else
        parent_indices = map(parent_index_range, indices, loc, topo, halo)
        windowed_parent = view(parent(data), parent_indices...)
    end

    sz = size(grid)

    return offset_data(windowed_parent, loc, topo, sz, halo, indices)
end

"""
    view(f::Field, indices...)

Returns a `Field` with `indices`, whose `data` is
a view into `f`, offset to preserve index meaning.

Example
=======

```@meta
DocTestSetup = quote
   using Random
   Random.seed!(1234)
end
```

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 3, 4), x=(0, 1), y=(0, 1), z=(0, 1));

julia> c = CenterField(grid)
2×3×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 8×9×10 OffsetArray(::Array{Float64, 3}, -2:5, -2:6, -2:7) with eltype Float64 with indices -2:5×-2:6×-2:7
    └── max=0.0, min=0.0, mean=0.0

julia> c .= rand(size(c)...);

julia> v = view(c, :, 2:3, 1:2)
2×2×2 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Nothing, north: Nothing, bottom: Nothing, top: Nothing, immersed: ZeroFlux
└── data: 8×2×2 OffsetArray(view(::Array{Float64, 3}, :, 5:6, 4:5), -2:5, 2:3, 1:2) with eltype Float64 with indices -2:5×2:3×1:2
    └── max=0.972136, min=0.0149088, mean=0.59198

julia> size(v)
(2, 2, 2)

julia> v[2, 2, 2] == c[2, 2, 2]
true
```
"""
function Base.view(f::Field, i, j, k)
    grid = f.grid
    loc = location(f)

    # Validate indices (convert Int to UnitRange, error for invalid indices)
    window_indices = validate_indices((i, j, k), loc, f.grid)
    
    # Choice: OffsetArray of view of OffsetArray, or OffsetArray of view?
    #     -> the first retains a reference to the original f.data (an OffsetArray)
    #     -> the second loses it, so we'd have to "re-offset" the underlying data to access.
    #     -> we choose the second here, opting to "reduce indirection" at the cost of "index recomputation".
    #
    # OffsetArray around a view of parent with appropriate indices:
    windowed_data = offset_windowed_data(f.data, loc, grid, window_indices)  

    boundary_conditions = FieldBoundaryConditions(window_indices, f.boundary_conditions)

    # "Sliced" Fields created here share data with their parent.
    # Therefore we set status=nothing so we don't conflate computation
    # of the sliced field with computation of the parent field.
    status = nothing

    return Field(loc,
                 grid,
                 windowed_data,
                 boundary_conditions,
                 window_indices,
                 f.operand,
                 status)
end

const WindowedData = OffsetArray{<:Any, <:Any, <:SubArray}
const WindowedField = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:WindowedData}

# Conveniences
Base.view(f::Field, I::Vararg{Colon}) = f
Base.view(f::Field, i) = view(f, i, :, :)
Base.view(f::Field, i, j) = view(f, i, j, :)

boundary_conditions(not_field) = nothing

function boundary_conditions(f::Field)
    if f.indices === default_indices(3) # default boundary conditions
        return f.boundary_conditions
    else # filter boundary conditions in windowed directions
        return FieldBoundaryConditions(f.indices, f.boundary_conditions)
    end
end

immersed_boundary_condition(f::Field) = f.boundary_conditions.immersed
data(field::Field) = field.data
indices(obj, i=default_indices(3)) = i
indices(f::Field, i=default_indices(3)) = f.indices
indices(a::SubArray, i=default_indices(ndims(a))) = a.indices
indices(a::OffsetArray, i=default_indices(ndims(a))) = indices(parent(a), i)

"""Return indices that create a `view` over the interior of a Field."""
interior_view_indices(field_indices, interior_indices) = Colon()
interior_view_indices(::Colon,       interior_indices) = interior_indices

instantiate(T::Type) = T()
instantiate(t) = t

function interior(a::OffsetArray,
                  Loc::Tuple,
                  Topo::Tuple,
                  sz::NTuple{N, Int},
                  halo_sz::NTuple{N, Int},
                  ind::Tuple=default_indices(3)) where N

    loc = map(instantiate, Loc)
    topo = map(instantiate, Topo)
    i_interior = map(interior_parent_indices, loc, topo, sz, halo_sz)
    i_view = map(interior_view_indices, ind, i_interior)
    return view(parent(a), i_view...)
end

"""
    interior(f::Field)

Returns a view of `f` that excludes halo points."
"""
interior(f::Field) = interior(f.data, location(f), f.grid, f.indices)
interior(a::OffsetArray, loc, grid, indices) = interior(a, loc, topology(grid), size(grid), halo_size(grid), indices)
interior(f::Field, I...) = view(interior(f), I...)
    
# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::Field, I...) = Base.checkbounds(f.data, I...)

function Base.axes(f::Field)
    if f.indices === (:, : ,:)
        return Base.OneTo.(size(f))
    else
        return Tuple(f.indices[i] isa Colon ? Base.OneTo(size(f, i)) : f.indices[i] for i = 1:3)
    end
end

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
@propagate_inbounds Base.getindex(f::Field, i::Int)  = parent(f)[i]
@propagate_inbounds Base.setindex!(f::Field, val, i, j, k) = setindex!(f.data, val, i, j, k)
@propagate_inbounds Base.lastindex(f::Field) = lastindex(f.data)
@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(f.data, dim)

Base.fill!(f::Field, val) = fill!(parent(f), val)
Base.parent(f::Field) = parent(f.data)
Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)

total_size(f::Field) = total_size(f.grid, location(f), f.indices)
Base.size(f::Field)  = size(f.grid, location(f), f.indices)

==(f::Field, a) = interior(f) == a
==(a, f::Field) = a == interior(f)
==(a::Field, b::Field) = interior(a) == interior(b)

#####
##### Interface for field computations
#####

"""
    compute!(field)

Computes `field.data` from `field.operand`.
"""
compute!(field, time=nothing) = field # fallback

"""
    @compute(exprs...)

Call `compute!` on fields after defining them.
"""
macro compute(def)
    expr = Expr(:block)
    field = def.args[1]
    push!(expr.args, :($(esc(def))))
    push!(expr.args, :(compute!($(esc(field)))))
    return expr
end

# Computation "status" for avoiding unnecessary recomputation
mutable struct FieldStatus{T}
    time :: T
end

FieldStatus() = FieldStatus(0.0)
Adapt.adapt_structure(to, status::FieldStatus) = (; time = status.time)

"""
    compute_at!(field, time)

Computes `field.data` at `time`. Falls back to compute!(field).
"""
compute_at!(field, time) = compute!(field)

"""
    compute_at!(field, time)

Computes `field.data` if `time != field.status.time`.
"""
function compute_at!(field::Field, time)
    if isnothing(field.status) # then always compute:
        compute!(field, time)

    # Otherwise, compute only on initialization or if field.status.time is not current,
    elseif time == zero(time) || time != field.status.time
        compute!(field, time)
        field.status.time = time
    end

    return field
end

# This edge case occurs if `fetch_output` is called with `model::Nothing`.
# We do the safe thing here and always compute.
compute_at!(field::Field, ::Nothing) = compute!(field, nothing)

#####
##### Fields that are reduced along one or more dimensions
#####

const XReducedField = Field{Nothing}
const YReducedField = Field{<:Any, Nothing}
const ZReducedField = Field{<:Any, <:Any, Nothing}

const YZReducedField = Field{<:Any, Nothing, Nothing}
const XZReducedField = Field{Nothing, <:Any, Nothing}
const XYReducedField = Field{Nothing, Nothing, <:Any}

const XYZReducedField = Field{Nothing, Nothing, Nothing}

const ReducedField = Union{XReducedField, YReducedField, ZReducedField,
                           YZReducedField, XZReducedField, XYReducedField,
                           XYZReducedField}

reduced_dimensions(field::Field)           = ()
reduced_dimensions(field::XReducedField)   = tuple(1)
reduced_dimensions(field::YReducedField)   = tuple(2)
reduced_dimensions(field::ZReducedField)   = tuple(3)
reduced_dimensions(field::YZReducedField)  = (2, 3)
reduced_dimensions(field::XZReducedField)  = (1, 3)
reduced_dimensions(field::XYReducedField)  = (1, 2)
reduced_dimensions(field::XYZReducedField) = (1, 2, 3)

@propagate_inbounds Base.getindex(r::XReducedField, i, j, k) = getindex(r.data, 1, j, k)
@propagate_inbounds Base.getindex(r::YReducedField, i, j, k) = getindex(r.data, i, 1, k)
@propagate_inbounds Base.getindex(r::ZReducedField, i, j, k) = getindex(r.data, i, j, 1)

@propagate_inbounds Base.setindex!(r::XReducedField, v, i, j, k) = setindex!(r.data, v, 1, j, k)
@propagate_inbounds Base.setindex!(r::YReducedField, v, i, j, k) = setindex!(r.data, v, i, 1, k)
@propagate_inbounds Base.setindex!(r::ZReducedField, v, i, j, k) = setindex!(r.data, v, i, j, 1)

@propagate_inbounds Base.getindex(r::YZReducedField, i, j, k) = getindex(r.data, i, 1, 1)
@propagate_inbounds Base.getindex(r::XZReducedField, i, j, k) = getindex(r.data, 1, j, 1)
@propagate_inbounds Base.getindex(r::XYReducedField, i, j, k) = getindex(r.data, 1, 1, k)

@propagate_inbounds Base.setindex!(r::YZReducedField, v, i, j, k) = setindex!(r.data, v, i, 1, 1)
@propagate_inbounds Base.setindex!(r::XZReducedField, v, i, j, k) = setindex!(r.data, v, 1, j, 1)
@propagate_inbounds Base.setindex!(r::XYReducedField, v, i, j, k) = setindex!(r.data, v, 1, 1, k)

@propagate_inbounds Base.getindex(r::XYZReducedField, i, j, k) = getindex(r.data, 1, 1, 1)
@propagate_inbounds Base.setindex!(r::XYZReducedField, v, i, j, k) = setindex!(r.data, v, 1, 1, 1)

# Preserve location when adapting fields reduced on one or more dimensions
function Adapt.adapt_structure(to, reduced_field::ReducedField)
    LX, LY, LZ = location(reduced_field)
    return Field{LX, LY, LZ}(nothing,
                             adapt(to, reduced_field.data),
                             nothing,
                             nothing,
                             nothing,
                             nothing,
                             nothing)
end

#####
##### Field reductions
#####

# TODO: needs test
Statistics.dot(a::Field, b::Field) = mapreduce((x, y) -> x * y, +, interior(a), interior(b))

# TODO: in-place allocations with function mappings need to be fixed in Julia Base...
const SumReduction     = typeof(Base.sum!)
const MeanReduction    = typeof(Statistics.mean!)
const ProdReduction    = typeof(Base.prod!)
const MaximumReduction = typeof(Base.maximum!)
const MinimumReduction = typeof(Base.minimum!)
const AllReduction     = typeof(Base.all!)
const AnyReduction     = typeof(Base.any!)

check_version_larger_than_7() = VERSION.minor > 7

initialize_reduced_field!(::SumReduction,  f, r::ReducedField, c) = check_version_larger_than_7() ? Base.initarray!(interior(r), f, Base.add_sum, true, interior(c))  : Base.initarray!(interior(r), Base.add_sum, true, interior(c))
initialize_reduced_field!(::ProdReduction, f, r::ReducedField, c) = check_version_larger_than_7() ? Base.initarray!(interior(r), f, Base.mul_prod, true, interior(c)) : Base.initarray!(interior(r), Base.mul_prod, true, interior(c))
initialize_reduced_field!(::AllReduction,  f, r::ReducedField, c) = check_version_larger_than_7() ? Base.initarray!(interior(r), f, &, true, interior(c))             : Base.initarray!(interior(r), &, true, interior(c))
initialize_reduced_field!(::AnyReduction,  f, r::ReducedField, c) = check_version_larger_than_7() ? Base.initarray!(interior(r), f, |, true, interior(c))             : Base.initarray!(interior(r), |, true, interior(c))

initialize_reduced_field!(::MaximumReduction, f, r::ReducedField, c) = Base.mapfirst!(f, interior(r), interior(c))
initialize_reduced_field!(::MinimumReduction, f, r::ReducedField, c) = Base.mapfirst!(f, interior(r), interior(c))

filltype(f, c) = eltype(c)
filltype(::Union{AllReduction, AnyReduction}, grid) = Bool

function reduced_location(loc; dims)
    if dims isa Colon
        return (Nothing, Nothing, Nothing)
    else
        return Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)
    end
end

reduced_indices(indices; dims) = Tuple(i ∈ dims ? Colon() : indices[i] for i in 1:3)

function reduced_dimension(loc)
    dims = ()
    for i in 1:3
        loc[i] == Nothing ? dims = (dims..., i) : dims
    end
    return dims
end

## Allow support for ConditionalOperation

get_neutral_mask(::Union{AllReduction, AnyReduction})  = true
get_neutral_mask(::Union{SumReduction, MeanReduction}) =   0
get_neutral_mask(::MinimumReduction) =   Inf
get_neutral_mask(::MaximumReduction) = - Inf
get_neutral_mask(::ProdReduction)    =   1

# If func = identity and condition = nothing, nothing happens
"""
    condition_operand(f::Function, op::AbstractField, condition, mask)

Wrap `f(op)` in `ConditionedOperand` with `condition` and `mask`. `f` defaults to `identity`.

If `f isa identity` and `isnothing(condition)` then `op` is returned without wrapping.

Otherwise return `ConditionedOperand`, even when `isnothing(condition)` but `!(f isa identity)`.
"""
@inline condition_operand(op::AbstractField, condition, mask) = condition_operand(identity, op, condition, mask)
@inline condition_operand(::typeof(identity), operand::AbstractField, ::Nothing, mask) = operand

@inline conditional_length(c::AbstractField)        = length(c)
@inline conditional_length(c::AbstractField, dims)  = mapreduce(i -> size(c, i), *, unique(dims); init=1)

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin
        
        # In-place
        function Base.$(reduction!)(f::Function,
                                    r::ReducedField,
                                    a::AbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            return Base.$(reduction!)(identity,
                                      interior(r),
                                      condition_operand(f, a, condition, mask);
                                      kwargs...)
        end

        function Base.$(reduction!)(r::ReducedField,
                                    a::AbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            return Base.$(reduction!)(identity,
                                      interior(r),
                                      condition_operand(a, condition, mask);
                                      kwargs...)
        end

        # Allocating
        function Base.$(reduction)(f::Function,
                                   c::AbstractField;
                                   condition = nothing,
                                   mask = get_neutral_mask(Base.$(reduction!)),
                                   dims = :)

            T = filltype(Base.$(reduction!), c)
            loc = reduced_location(location(c); dims)
            r = Field(loc, c.grid, T; indices=indices(c))
            conditioned_c = condition_operand(f, c, condition, mask)
            initialize_reduced_field!(Base.$(reduction!), identity, r, conditioned_c)
            Base.$(reduction!)(identity, r, conditioned_c, init=false)

            if dims isa Colon
                return CUDA.@allowscalar first(r)
            else
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; kwargs...) = Base.$(reduction)(identity, c; kwargs...)
    end
end

function Statistics._mean(f, c::AbstractField, ::Colon; condition = nothing, mask = 0) 
    operator = condition_operand(f, c, condition, mask)
    return sum(operator) / conditional_length(operator)
end

function Statistics._mean(f, c::AbstractField, dims; condition = nothing, mask = 0)
    operand = condition_operand(f, c, condition, mask)
    r = sum(operand; dims)
    n = conditional_length(operand, dims)
    r ./= n
    return r
end

Statistics.mean(f::Function, c::AbstractField; condition = nothing, dims=:) = Statistics._mean(f, c, dims; condition)
Statistics.mean(c::AbstractField; condition = nothing, dims=:) = Statistics._mean(identity, c, dims; condition)

function Statistics.mean!(f::Function, r::ReducedField, a::AbstractField; condition = nothing, mask = 0)
    sum!(f, r, a; condition, mask, init=true)
    dims = reduced_dimension(location(r))
    n = conditional_length(condition_operand(f, a, condition, mask), dims)
    r ./= n
    return r
end

Statistics.mean!(r::ReducedField, a::AbstractArray; kwargs...) = Statistics.mean!(identity, r, a; kwargs...)

function Statistics.norm(a::AbstractField; condition = nothing)
    r = zeros(a.grid, 1)
    Base.mapreducedim!(x -> x * x, +, r, condition_operand(a, condition, 0))
    return CUDA.@allowscalar sqrt(r[1])
end

function Base.isapprox(a::AbstractField, b::AbstractField; kw...)
    conditioned_a = condition_operand(a, nothing, one(eltype(a)))
    conditioned_b = condition_operand(b, nothing, one(eltype(b)))
    # TODO: Make this non-allocating?
    return all(isapprox.(conditioned_a, conditioned_b; kw...))
end

#####
##### fill_halo_regions!
#####

function fill_halo_regions!(field::Field, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    # To correctly fill the halo regions of fields with non-default indices, we'd have to
    # offset indices in the fill halo regions kernels.
    # For now we punt and don't support filling halo regions on windowed fields.
    # Note that `FieldBoundaryConditions` _can_ filter boundary conditions in
    # windowed directions:
    #
    #   filtered_bcs = FieldBoundaryConditions(field.indices, field.boundary_conditions)
    #  
    # which will be useful for implementing halo filling for windowed fields in the future.
    fill_halo_regions!(field.data,
                       field.boundary_conditions,
                       field.indices,
                       instantiated_location(field),
                       field.grid,
                       args...;
                       reduced_dimensions = reduced_dims,
                       kwargs...)

    return nothing
end


using Oceananigans.Architectures: device_event
using Oceananigans.BoundaryConditions: OBC

using Adapt
using KernelAbstractions: @kernel, @index
using Base: @propagate_inbounds

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Statistics: norm, mean, mean!

struct Field{LX, LY, LZ, O, G, T, D, B, S} <: AbstractField{LX, LY, LZ, G, T, 3}
    grid :: G
    data :: D
    boundary_conditions :: B
    operand :: O
    status :: S

    # Inner constructor that does not validate _anything_!
    function Field{LX, LY, LZ}(grid::G, data::D, bcs::B, op::O, status::S) where {LX, LY, LZ, G, D, B, O, S}
        T = eltype(data)
        return new{LX, LY, LZ, O, G, T, D, B, S}(grid, data, bcs, op, status)
    end
end

function validate_field_data(loc, data, grid)
    Tx, Ty, Tz = total_size(loc, grid)

    if size(data) != (Tx, Ty, Tz)
        LX, LY, LZ = loc    
        e = "Cannot construct field at ($LX, $LY, $LZ) with size(data)=$(size(data)). " *
            "`data` must have size ($Tx, $Ty, $Tz)."
        throw(ArgumentError(e))
    end

    return nothing
end

validate_boundary_condition_location(bc, ::Center, side) = nothing                  # anything goes for centers
validate_boundary_condition_location(::Union{OBC, Nothing}, ::Face, side) = nothing # only open or nothing on faces
validate_boundary_condition_location(::Nothing, ::Nothing, side) = nothing          # its nothing or nothing
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

# Common outer constructor for all field flavors that validates data and boundary conditions
function Field(loc::Tuple, grid::AbstractGrid, data, bcs, op, status)
    validate_field_data(loc, data, grid)
    validate_boundary_conditions(loc, grid, bcs)
    LX, LY, LZ = loc
    return Field{LX, LY, LZ}(grid, data, bcs, op, status)
end

#####
##### Vanilla, non-computed Field
#####

"""
    Field{LX, LY, LZ}(grid; kw...)

Construct a `Field` on `grid` at the location `(LX, LY, LZ)`.
Each of `(LX, LY, LZ)` is either `Center` or `Face` and determines
the field's location in `(x, y, z)`.

Keyword arguments
=================

- data: 

Example
=======

```jldoctest
julia> using Oceananigans

julia> ω = Field{Face, Face, Center}(RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))
1×1×1 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── boundary conditions: west=Periodic, east=Periodic, south=Periodic, north=Periodic, bottom=ZeroFlux, top=ZeroFlux, immersed=ZeroFlux
└── data: 3×3×3 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2
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
               data = new_data(T, grid, loc),
               boundary_conditions = FieldBoundaryConditions(grid, loc))

    return Field(loc, grid, data, boundary_conditions, nothing, nothing)
end
    
Field(f::Field) = f # indeed
Field(z::ZeroField) = z

#####
##### Field utils
#####

# Canonical `similar` for Field (doesn't transfer boundary conditions)
function Base.similar(f::Field, grid=f.grid)
    loc = location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(parent(f)), grid, loc),
                 FieldBoundaryConditions(grid, loc),
                 f.operand,
                 deepcopy(f.status))
end

# Fallback: cannot infer boundary conditions.
boundary_conditions(field) = nothing
boundary_conditions(f::Field) = f.boundary_conditions

function interior(a::Union{Field, OffsetArray}, (LX, LY, LZ), grid)
    TX, TY, TZ = topology(grid)
    ii = interior_parent_indices(LX, TX, grid.Nx, grid.Hx)
    jj = interior_parent_indices(LY, TY, grid.Ny, grid.Hy)
    kk = interior_parent_indices(LZ, TZ, grid.Nz, grid.Hz)
    return view(parent(a), ii, jj, kk)
end

"Returns a view of `f` that excludes halo points."
interior(f::Field) = interior(f, location(f), f.grid)
    
function interior_copy(f::Field)
    LX, LY, LZ = location(f)
    return parent(f)[interior_parent_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx),
                     interior_parent_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy),
                     interior_parent_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz)]
end

# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::Field, I...) = Base.checkbounds(f.data, I...)

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
@propagate_inbounds Base.getindex(f::Field, i::Int)  = parent(f)[i]
@propagate_inbounds Base.setindex!(f::Field, val, i, j, k) = setindex!(f.data, val, i, j, k)
@propagate_inbounds Base.lastindex(f::Field) = lastindex(f.data)
@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(f.data, dim)

Base.fill!(f::Field, val) = fill!(parent(f), val)
Base.isapprox(ϕ::Field, ψ::Field; kw...) = isapprox(interior(ϕ), interior(ψ); kw...)
Base.parent(f::Field) = parent(f.data)
Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)

data(f::Field) = f.data
cpudata(a::Field) = arch_array(CPU(), a.data)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CenterField(grid; kwargs...)

Returns `Field{Center, Center, Center}` on `arch`itecture and `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
CenterField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field{Center, Center, Center}(grid, T; kw...)

"""
    XFaceField(grid; kw...)

Returns `Field{Face, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
XFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field{Face, Center, Center}(grid, T; kw...)

"""
    YFaceField(grid; kw...)

Returns `Field{Center, Face, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
YFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field{Center, Face, Center}(grid, T; kw...)

"""
    ZFaceField(grid; kw...)

Returns `Field{Center, Center, Face}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
ZFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field{Center, Center, Face}(grid, T; kw...)

#####
##### Interface for field computations
#####

"""
    compute!(field)

Computes `field.data` from `field.operand`.
"""
compute!(field, time=nothing) = nothing # fallback

"""
    @compute(exprs...)

Call compute! on fields after defining them.
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

reduced_dimensions(field::Field) = ()

reduced_dimensions(field::XReducedField) = tuple(1)
reduced_dimensions(field::YReducedField) = tuple(2)
reduced_dimensions(field::ZReducedField) = tuple(3)

reduced_dimensions(field::YZReducedField) = (2, 3)
reduced_dimensions(field::XZReducedField) = (1, 3)
reduced_dimensions(field::XYReducedField) = (1, 2)

reduced_dimensions(field::XYZReducedField) = (1, 2, 3)

function fill_halo_regions!(field::Field, arch, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)
    if reduced_dimensions === () # the field is not reduced!
        return fill_halo_regions!(field.data,
                                  field.boundary_conditions,
                                  architecture(field),
                                  field.grid,
                                  args...;
                                  kwargs...)
    else
        return fill_halo_regions!(field.data,
                                  field.boundary_conditions,
                                  architecture(field),
                                  field.grid,
                                  args...;
                                  reduced_dimensions=reduced_dims,
                                  kwargs...)
    end
end

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

initialize_reduced_field!(::SumReduction,  f, r::ReducedField, c) = Base.initarray!(interior(r), Base.add_sum, true, interior(c))
initialize_reduced_field!(::ProdReduction, f, r::ReducedField, c) = Base.initarray!(interior(r), Base.mul_prod, true, interior(c))
initialize_reduced_field!(::AllReduction,  f, r::ReducedField, c) = Base.initarray!(interior(r), &, true, interior(c))
initialize_reduced_field!(::AnyReduction,  f, r::ReducedField, c) = Base.initarray!(interior(r), |, true, interior(c))

initialize_reduced_field!(::MaximumReduction, f, r::ReducedField, c) = Base.mapfirst!(f, interior(r), c)
initialize_reduced_field!(::MinimumReduction, f, r::ReducedField, c) = Base.mapfirst!(f, interior(r), c)

filltype(f, c) = eltype(c)
filltype(::Union{AllReduction, AnyReduction}, grid) = Bool

function reduced_location(loc; dims)
    if dims isa Colon
        return (Nothing, Nothing, Nothing)
    else
        return Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)
    end
end

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

@inline condition_operand(operand, ::Nothing, mask)                = operand
@inline condition_operand(operand::AbstractField, ::Nothing, mask) = operand

@inline conditional_length(c::AbstractField)        = length(c)
@inline conditional_length(c::AbstractField, dims)  = mapreduce(i -> size(c, i), *, unique(dims); init=1)

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin
        
        # In-place
        Base.$(reduction!)(f::Function, r::ReducedField, a::AbstractArray;
                           condition = nothing, mask = get_neutral_mask(Base.$(reduction!)), kwargs...) =
            Base.$(reduction!)(f, interior(r), condition_operand(a, condition, mask); kwargs...)

        Base.$(reduction!)(r::ReducedField, a::AbstractArray; 
                           condition = nothing, mask = get_neutral_mask(Base.$(reduction!)), kwargs...) =
            Base.$(reduction!)(identity, interior(r), condition_operand(a, condition, mask); kwargs...)

        # Allocating
        function Base.$(reduction)(f::Function, c::AbstractField;
                                   condition = nothing, mask = get_neutral_mask(Base.$(reduction!)),
                                   dims=:)
            if dims isa Colon
                r = zeros(architecture(c), c.grid, 1, 1, 1)
                Base.$(reduction!)(f, r, condition_operand(c, condition, mask))
                return CUDA.@allowscalar r[1, 1, 1]
            else
                T = filltype(Base.$(reduction!), c)
                loc = reduced_location(location(c); dims)
                r = Field(loc, c.grid, T)
                initialize_reduced_field!(Base.$(reduction!), f, r, condition_operand(c, condition, mask))
                Base.$(reduction!)(f, r, condition_operand(c, condition, mask), init=false)
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; kwargs...) = Base.$(reduction)(identity, c; kwargs...)
    end
end

function Statistics._mean(f, c::AbstractField, ::Colon; condition = nothing, mask = 0) 
    operator = condition_operand(c, condition, mask)
    return sum(f, operator) / conditional_length(operator)
end

function Statistics._mean(f, c::AbstractField, dims; condition = nothing, mask = 0)
    operator = condition_operand(c, condition, mask)
    r = sum(f, operator; dims)
    n = conditional_length(operator, dims)
    r ./= n
    return r
end

Statistics.mean(f::Function, c::AbstractField; condition = nothing, dims=:) = Statistics._mean(f, c, dims; condition)
Statistics.mean(c::AbstractField; condition = nothing, dims=:) = Statistics._mean(identity, c, dims; condition)

function Statistics.mean!(f::Function, r::ReducedField, a::AbstractArray; condition = nothing, mask = 0)
    sum!(f, r, a; condition, mask, init=true)
    dims = reduced_dimension(location(r))
    n = conditional_length(condition_operand(a, condition, mask), dims)
    r ./= n
    return r
end

Statistics.mean!(r::ReducedField, a::AbstractArray; kwargs...) = Statistics.mean!(identity, r, a; kwargs...)

function Statistics.norm(a::AbstractField; condition = nothing)
    r = zeros(a.grid, 1)
    Base.mapreducedim!(x -> x * x, +, r, condition_operand(a, condition, 0))
    return CUDA.@allowscalar sqrt(r[1])
end
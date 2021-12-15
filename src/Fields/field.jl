using Oceananigans.Architectures: device_event

using Adapt
using KernelAbstractions: @kernel, @index

using Base: @propagate_inbounds

struct Field{LX, LY, LZ, O, A, G, T, D, B, S} <: AbstractField{LX, LY, LZ, A, G, T, 3}
    grid :: G
    data :: D
    boundary_conditions :: B
    operand :: O
    status :: S

    # Inner constructor that does not validate _anything_!
    function Field{LX, LY, LZ}(grid::G, data::D, bcs::B, op::O, status::S) where {LX, LY, LZ, G, D, B, O, S}
        T = eltype(grid)
        A = typeof(architecture(grid))
        return new{LX, LY, LZ, O, A, G, T, D, B, S}(grid, data, bcs, op, status)
    end
end

# Common outer constructor for all field flavors that validates data and boundary conditions
function Field(loc::Tuple, grid::AbstractGrid, data, bcs, op, status)
    validate_field_data(loc, data, grid)
    # validate_boundary_conditions(loc, grid, bcs)
    arch = architecture(grid)
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
Field located at (Face, Face, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 1)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
└── boundary conditions: west=Periodic, east=Periodic, south=Periodic, north=Periodic, bottom=ZeroFlux, top=ZeroFlux, immersed=ZeroFlux
```
"""
Field{LX, LY, LZ}(grid::AbstractGrid, T::DataType=eltype(grid); kw...) where {LX, LY, LZ} =
    Field((LX, LY, LZ), grid, T; kw...)

function Field(loc::Tuple,
               grid::AbstractGrid,
               T::DataType = eltype(grid);
               data = new_data(T, grid, loc),
               boundary_conditions = FieldBoundaryConditions(grid, loc))

    return Field(loc, grid, data, boundary_conditions, nothing, nothing)
end

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

data(f::Field) = f.data

"Returns a view of `f` that excludes halo points."
function interior(f::Field)
    LX, LY, LZ = location(f)
    TX, TY, TZ = topology(f.grid)
    ii = interior_parent_indices(LX, TX, f.grid.Nx, f.grid.Hx)
    jj = interior_parent_indices(LY, TY, f.grid.Ny, f.grid.Hy)
    kk = interior_parent_indices(LZ, TZ, f.grid.Nz, f.grid.Hz)
    return view(parent(f), ii, jj, kk)
end

# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::Field, I...) = Base.checkbounds(data(f), I...)
@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(data(f), inds...)
@propagate_inbounds Base.getindex(f::Field, i::Int)  = parent(f)[i]
@propagate_inbounds Base.setindex!(f::Field, val, i, j, k) = setindex!(data(f), val, i, j, k)
@propagate_inbounds Base.lastindex(f::Field) = lastindex(data(f))
@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(data(f), dim)
Base.fill!(f::Field, val) = fill!(parent(f), val)

Base.isapprox(ϕ::Field, ψ::Field; kw...) = isapprox(interior(ϕ), interior(ψ); kw...)

Adapt.adapt_structure(to, field::Field) = Adapt.adapt(to, field.data)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CenterField(grid; kwargs...)

Returns `Field{Center, Center, Center}` on `arch`itecture and `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
CenterField(grid::AbstractGrid, FT=eltype(grid); kw...) = Field{Center, Center, Center}(grid, FT; kw...)

"""
    XFaceField(grid; kw...)

Returns `Field{Face, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
XFaceField(grid::AbstractGrid, FT=eltype(grid); kw...) = Field{Face, Center, Center}(grid, FT; kw...)

"""
    YFaceField(grid; kw...)

Returns `Field{Center, Face, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
YFaceField(grid::AbstractGrid, FT=eltype(grid); kw...) = Field{Center, Face, Center}(grid, FT; kw...)

"""
    ZFaceField(grid; kw...)

Returns `Field{Center, Center, Face}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
ZFaceField(grid::AbstractGrid, FT=eltype(grid); kw...) = Field{Center, Center, Face}(grid, FT; kw...)

#####
##### Fields computed from AbstractOperation and associated utilities
#####

const ComputedField = Field{<:Any, <:Any, <:Any, <:AbstractOperation}

"""
    compute!(field)

Computes `field.data` from `field.operand`.
"""
compute!(field) = nothing # fallback

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

"""
    compute_at!(field, time)

Computes `field.data` at `time`. Falls back to compute!(field).
"""
compute_at!(field, time) = compute!(field)

"""
    conditional_compute!(field, time)

Computes `field.data` if `time != field.status.time`.
"""
function conditional_compute!(field, time)
    if time == zero(time) || time != field.status.time
        compute!(field, time)
        field.status.time = time
    end
    return nothing
end

# This edge case occurs if `fetch_output` is called with `model::Nothing`.
# We do the safe thing here and always compute.
conditional_compute!(field, ::Nothing) = compute!(field, nothing)

mutable struct FieldStatus{T}
    time :: T
end

Adapt.adapt_structure(to, status::FieldStatus) = (; time = status.time)

"""
    Field(operand::AbstractOperation; kwargs...)

Return `f::Field` where `f.data` is computed from `f.operand` by
calling compute!(f).

Keyword arguments
=================

data (AbstractArray): An offset Array or CuArray for storing the result of a computation.
                      Must have `total_size(location(operand), grid)`.

boundary_conditions (FieldBoundaryConditions): Boundary conditions for `f`. 

recompute_safely (Bool): whether or not to _always_ "recompute" `f` if `f` is
                         nested within another computation via an `AbstractOperation`.
                         If `data` is not provided then `recompute_safely=false` and
                         recomputation is _avoided_. If `data` is provided, then
                         `recompute_safely=true` by default.
"""
function Field(operand::AbstractOperation;
               data = nothing,
               boundary_conditions = FieldBoundaryConditions(op.grid, location(op)),
               recompute_safely = true)

    if isnothing(data)
        data = new_data(op.grid, location(op))
        recompute_safely = false
    end

    status = recompute_safely ? nothing : FieldStatus(0.0)

    return Field(location(op), op.grid, data, boundary_conditions, operand, status)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    workgroup, worksize =
        work_layout(comp.grid, :xyz, include_right_boundaries = true, location = location(comp))

    arch = architecture(comp)
    compute_kernel! = _compute!(device(arch), workgroup, worksize)
    event = compute_kernel!(comp.data, comp.operand; dependencies = device_event(arch))
    wait(device(arch), event)

    fill_halo_regions!(comp, arch)

    return comp
end

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = operand[i, j, k]
end

function compute_at!(field::ComputedField, time)
    if isnothing(field.status)
        return compute!(field, time)
    else
        return conditional_compute!(field, time)
    end
end

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
                             nothing, nothing, nothing)
end

#####
##### Field reductions
#####

# Risky to use these without tests. Docs would also be nice.
Statistics.dot(a::Field, b::Field) = mapreduce((x, y) -> x * y, +, interior(a), interior(b))

# TODO: In-place allocations with function mappings need to be fixed in Julia Base...
const SumReduction     = typeof(Base.sum!)
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

filltype(f, grid) = eltype(grid)
filltype(::Union{AllReduction, AnyReduction}, grid) = Bool

reduced_location(loc; dims) = Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any)

    reduction! = Symbol(reduction, '!')

    @eval begin

        # In-place
        Base.$(reduction!)(f::Function, r::ReducedField, a::AbstractArray; kwargs...) =
            Base.$(reduction!)(f, interior(r), a; kwargs...)

        Base.$(reduction!)(r::ReducedField, a::AbstractArray; kwargs...) =
            Base.$(reduction!)(identity, interior(r), a; kwargs...)

        # Allocating
        function Base.$(reduction)(f::Function, c::AbstractField; dims=:)
            if dims === (:)
                r = zeros(architecture(c), c.grid, 1, 1, 1)
                Base.$(reduction!)(f, r, c)
                return CUDA.@allowscalar r[1, 1, 1]
            else
                T = filltype(Base.$(reduction!), c.grid)
                loc = reduced_location(location(c); dims)
                r = Field(loc, c.grid, T)
                initialize_reduced_field!(Base.$(reduction!), f, r, c)
                Base.$(reduction!)(f, r, c, init=false)
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; dims=:) = Base.$(reduction)(identity, c; dims)
    end
end

Statistics._mean(f, c::AbstractField, ::Colon) = sum(f, c) / length(c)

function Statistics._mean(f, c::AbstractField, dims)
    r = sum(f, c; dims)
    n = mapreduce(i -> size(c, i), *, unique(dims); init=1)
    parent(r) ./= n
    return r
end


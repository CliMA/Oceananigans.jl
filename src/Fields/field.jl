using Oceananigans.BoundaryConditions: OBC, MCBC, BoundaryCondition, Zipper, construct_boundary_conditions_kernels
using Oceananigans.Grids: parent_index_range, index_range_offset, default_indices, all_indices, validate_indices
using Oceananigans.Grids: index_range_contains
using Oceananigans.Architectures: convert_to_device

using Adapt
using LinearAlgebra
using KernelAbstractions: @kernel, @index
using Base: @propagate_inbounds
using GPUArraysCore: @allowscalar

import Oceananigans: boundary_conditions
import Oceananigans.Architectures: on_architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!, getbc
import Statistics: mean, mean!
import LinearAlgebra: dot, norm
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
    communication_buffers :: F

    # Inner constructor that does not validate _anything_!
    function Field{LX, LY, LZ}(grid::G, data::D, bcs::B, indices::I, op::O, status::S, buffers::F) where {LX, LY, LZ, G, D, B, O, S, I, F}
        T = eltype(data)
        @apply_regionally new_bcs = construct_boundary_conditions_kernels(bcs, data, grid, (LX(), LY(), LZ()), indices) # Adding the kernels to the bcs
        return new{LX, LY, LZ, O, G, I, D, T, typeof(new_bcs), S, F}(grid, data, new_bcs, indices, op, status, buffers)
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
        ℓ = loc[dir]
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

# Some special validation for a zipper boundary condition
validate_boundary_condition_location(bc::Zipper, loc::Center, side) =
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))

validate_boundary_condition_location(bc::Zipper, loc::Face, side) =
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))


#####
##### Some basic constructors
#####

# Common outer constructor for all field flavors that performs input validation
function Field(loc::Tuple{<:LX, <:LY, <:LZ}, grid::AbstractGrid, data, bcs, indices, op=nothing, status=nothing) where {LX, LY, LZ}
    @apply_regionally indices = validate_indices(indices, loc, grid)
    @apply_regionally validate_field_data(loc, data, grid, indices)
    @apply_regionally validate_boundary_conditions(loc, grid, bcs)
    buffers = communication_buffers(grid, data, bcs)
    return Field{LX, LY, LZ}(grid, data, bcs, indices, op, status, buffers)
end

# Allocator for buffers used in fields that require ``communication''
# Extended in the `DistributedComputations` and the `MultiRegion` module
communication_buffers(grid, data, bcs) = nothing

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
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 6×9×10 OffsetArray(::Array{Float64, 3}, -1:4, -2:6, -2:7) with eltype Float64 with indices -1:4×-2:6×-2:7
    └── max=0.0, min=0.0, mean=0.0
```

Now, using `indices` we can create a two dimensional ``x``-``y`` field at location
`(Face, Face, Center)` to compute, e.g., the vertical vorticity ``∂v/∂x - ∂u/∂y``
at the fluid's surface ``z = 0``, which for `Center` corresponds to `k = Nz`.

```jldoctest fields
julia> u = XFaceField(grid); v = YFaceField(grid);

julia> ωₛ = Field(∂x(v) - ∂y(u), indices=(:, :, grid.Nz))
2×3×1 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: Nothing
├── indices: (:, :, 4:4)
├── operand: BinaryOperation at (Face, Face, Center)
├── status: time=0.0
└── data: 6×9×1 OffsetArray(::Array{Float64, 3}, -1:4, -2:6, 4:4) with eltype Float64 with indices -1:4×-2:6×4:4
    └── max=0.0, min=0.0, mean=0.0

julia> compute!(ωₛ)
2×3×1 Field{Face, Face, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: Nothing
├── indices: (:, :, 4:4)
├── operand: BinaryOperation at (Face, Face, Center)
├── status: time=0.0
└── data: 6×9×1 OffsetArray(::Array{Float64, 3}, -1:4, -2:6, 4:4) with eltype Float64 with indices -1:4×-2:6×4:4
    └── max=0.0, min=0.0, mean=0.0
```
"""
function Field{LX, LY, LZ}(grid::AbstractGrid,
                           T::DataType=eltype(grid);
                           kw...) where {LX, LY, LZ}

    return Field((LX(), LY(), LZ()), grid, T; kw...)
end

function Field(loc::Tuple, # These are instantiated locations, e.g. (Center(), Face(), nothing)
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
    CenterField(grid, T=eltype(grid); kw...)

Return a `Field{Center, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
CenterField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center(), Center(), Center()), grid, T; kw...)

"""
    XFaceField(grid, T=eltype(grid); kw...)

Return a `Field{Face, Center, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
XFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Face(), Center(), Center()), grid, T; kw...)

"""
    YFaceField(grid, T=eltype(grid); kw...)

Return a `Field{Center, Face, Center}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
YFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center(), Face(), Center()), grid, T; kw...)

"""
    ZFaceField(grid, T=eltype(grid); kw...)

Return a `Field{Center, Center, Face}` on `grid`.
Additional keyword arguments are passed to the `Field` constructor.
"""
ZFaceField(grid::AbstractGrid, T::DataType=eltype(grid); kw...) = Field((Center(), Center(), Face()), grid, T; kw...)

#####
##### Field utils
#####

# Canonical `similar` for Field (doesn't transfer boundary conditions)
function Base.similar(f::Field, grid=f.grid)
    loc = instantiated_location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(grid), grid, loc, f.indices),
                 FieldBoundaryConditions(grid, loc, f.indices),
                 f.indices,
                 f.operand,
                 deepcopy(f.status))
end

"""
    offset_windowed_data(data, data_indices, loc, grid, view_indices)

Return an `OffsetArray` of `parent(data)`.

If `indices` is not (:, :, :), a `view` of `parent(data)` with `indices`.

If `indices === (:, :, :)`, return an `OffsetArray` of `parent(data)`.
"""
function offset_windowed_data(data, data_indices, loc, grid, view_indices)
    halo = halo_size(grid)
    TX, TY, TZ = topology(grid)
    𝓉x = instantiate(TX)
    𝓉y = instantiate(TY)
    𝓉z = instantiate(TZ)

    topo = (𝓉x, 𝓉y, 𝓉z)
    parent_indices = parent_index_range.(data_indices, view_indices, loc, topo, halo)
    windowed_parent = view(parent(data), parent_indices...)

    sz = size(grid)
    return offset_data(windowed_parent, loc, topo, sz, halo, view_indices)
end

convert_colon_indices(view_indices, field_indices) = view_indices
convert_colon_indices(::Colon, field_indices) = field_indices
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

julia> c = CenterField(grid);

julia> set!(c, rand(size(c)...))
2×3×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 6×9×10 OffsetArray(::Array{Float64, 3}, -1:4, -2:6, -2:7) with eltype Float64 with indices -1:4×-2:6×-2:7
    └── max=0.972136, min=0.0149088, mean=0.626341

julia> v = view(c, :, 2:3, 1:2)
2×2×2 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Nothing, north: Nothing, bottom: Nothing, top: Nothing, immersed: Nothing
├── indices: (:, 2:3, 1:2)
└── data: 6×2×2 OffsetArray(view(::Array{Float64, 3}, :, 5:6, 4:5), -1:4, 2:3, 1:2) with eltype Float64 with indices -1:4×2:3×1:2
    └── max=0.972136, min=0.0149088, mean=0.59198

julia> size(v)
(2, 2, 2)

julia> v[2, 2, 2] == c[2, 2, 2]
true
```
"""
function Base.view(f::Field, i, j, k)
    grid = f.grid
    loc = instantiated_location(f)

    # Validate indices (convert Int to UnitRange, error for invalid indices)
    view_indices = validate_indices((i, j, k), loc, f.grid)

    if view_indices == f.indices # nothing to "view" here
        return f # we want the whole field after all.
    end

    # Check that the indices actually work here
    @apply_regionally valid_view_indices = map(index_range_contains, f.indices, view_indices)
    
    all(getregion(valid_view_indices, 1)) ||
        throw(ArgumentError("view indices $((i, j, k)) do not intersect field indices $(f.indices)"))
    
    @apply_regionally begin
        view_indices = map(convert_colon_indices, view_indices, f.indices)

        # Choice: OffsetArray of view of OffsetArray, or OffsetArray of view?
        #     -> the first retains a reference to the original f.data (an OffsetArray)
        #     -> the second loses it, so we'd have to "re-offset" the underlying data to access.
        #     -> we choose the second here, opting to "reduce indirection" at the cost of "index recomputation".
        #
        # OffsetArray around a view of parent with appropriate indices:
        windowed_data = offset_windowed_data(f.data, f.indices, loc, grid, view_indices)

        boundary_conditions = FieldBoundaryConditions(view_indices, f.boundary_conditions)
    end
    # "Sliced" Fields created here share data with their parent.
    # Therefore we set status=nothing so we don't conflate computation
    # of the sliced field with computation of the parent field.
    status = nothing

    return Field(loc,
                 grid,
                 windowed_data,
                 boundary_conditions,
                 view_indices,
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

@inline boundary_conditions(f::Field) = f.boundary_conditions
@inline boundary_conditions(w::WindowedField) = FieldBoundaryConditions(w.indices, w.boundary_conditions)

immersed_boundary_condition(f::Field) = f.boundary_conditions.immersed
data(field::Field) = field.data

instantiate(T::Type) = T()
instantiate(t) = t

# Heuristic for tuples
instantiate(T::Tuple{<:Type}) = (T[1]())
instantiate(T::Tuple{<:Type, <:Type}) = (T[1](), T[2]())
instantiate(T::Tuple{<:Type, <:Type, <:Type}) = (T[1](), T[2](), T[3]())
instantiate(T::Tuple{<:Type, <:Type, <:Type, <:Type}) = (T[1](), T[2](), T[3](), T[4]())
instantiate(T::NTuple{N, <:Type}) where N = map(instantiate, T)

"""Return indices that create a `view` over the interior of a Field."""
interior_view_indices(field_indices, interior_indices)   = Colon()
interior_view_indices(::Colon,       interior_indices)   = interior_indices

function interior(a::OffsetArray,
                  Loc::Tuple,
                  Topo::Tuple,
                  sz::NTuple{N, Int},
                  halo_sz::NTuple{N, Int},
                  ind::Tuple=default_indices(3)) where N

    ℓx, ℓy, ℓz = instantiate(Loc)
    𝓉x, 𝓉y, 𝓉z = instantiate(Topo)
    Nx, Ny, Nz = sz
    Hx, Hy, Hz = halo_sz
    i = interior_parent_indices(ℓx, 𝓉x, Nx, Hx)
    j = interior_parent_indices(ℓy, 𝓉y, Ny, Hy)
    k = interior_parent_indices(ℓz, 𝓉z, Nz, Hz)

    iv = @inbounds interior_view_indices(ind[1], i)
    jv = @inbounds interior_view_indices(ind[2], j)
    kv = @inbounds interior_view_indices(ind[3], k)

    return view(parent(a), iv, jv, kv)
end

"""
    interior(f::Field)

Return a view of `f` that excludes halo points.
"""
interior(f::Field) = interior(f.data, location(f), f.grid, f.indices)
interior(a::OffsetArray, loc, grid, indices) = interior(a, loc, topology(grid), size(grid), halo_size(grid), indices)
interior(f::Field, I...) = view(interior(f), I...)

# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::Field, I...) = Base.checkbounds(f.data, I...)

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
@propagate_inbounds Base.getindex(f::Field, i::Int)  = parent(f)[i]
@propagate_inbounds Base.setindex!(f::Field, val, i, j, k) = setindex!(f.data, val, i, j, k)
@propagate_inbounds Base.lastindex(f::Field) = lastindex(f.data)
@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(f.data, dim)

@inline Base.fill!(f::Field, val) = fill!(parent(f), val)
@inline Base.parent(f::Field) = parent(f.data)
Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)
Adapt.parent_type(::Type{<:Field{LX, LY, LZ, O, G, I, D}}) where {LX, LY, LZ, O, G, I, D} = D

total_size(f::Field) = total_size(f.grid, location(f), f.indices)
@inline Base.size(f::Field)  = size(f.grid, location(f), f.indices)

==(f::Field, a) = interior(f) == a
==(a, f::Field) = a == interior(f)

function ==(a::Field, b::Field)
    if architecture(a) == architecture(b)
        return interior(a) == interior(b)
    elseif architecture(a) isa CPU && architecture(b) isa GPU
        b_cpu = on_architecture(CPU(), b)
        return a == b_cpu
    elseif architecture(b) isa CPU && architecture(a) isa GPU
        a_cpu = on_architecture(CPU(), a)
        return a_cpu == b
    else
        throw(ArgumentError("Unable to assess the equality of \n $(summary(a)) \n \n versus \n \n $(summary(b))"))
    end
end

#####
##### Move Fields between architectures
#####

on_architecture(arch, field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(on_architecture(arch, field.grid),
                      on_architecture(arch, field.data),
                      on_architecture(arch, field.boundary_conditions),
                      on_architecture(arch, field.indices),
                      on_architecture(arch, field.operand),
                      on_architecture(arch, field.status),
                      on_architecture(arch, field.communication_buffers))

#####
##### Interface for field computations
#####

"""
    compute!(field)

Computes `field.data` from `field.operand`.
"""
compute!(field, time=nothing) = field # fallback

compute!(collection::Union{Tuple, NamedTuple}) = map(compute!, collection)

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
    FixedTime(time)

Represents a fixed compute time.
"""
struct FixedTime{T}
    time :: T
end

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
    if !(field.status isa FieldStatus) # then always compute:
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

const ReducedField = Union{XReducedField,
                           YReducedField,
                           ZReducedField,
                           YZReducedField,
                           XZReducedField,
                           XYReducedField,
                           XYZReducedField}

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

# Boundary conditions reduced in one direction --- drop boundary-normal index
@inline getbc(condition::XReducedField, j::Integer, k::Integer, grid::AbstractGrid, args...) = @inbounds condition[1, j, k]
@inline getbc(condition::YReducedField, i::Integer, k::Integer, grid::AbstractGrid, args...) = @inbounds condition[i, 1, k]
@inline getbc(condition::ZReducedField, i::Integer, j::Integer, grid::AbstractGrid, args...) = @inbounds condition[i, j, 1]

# Boundary conditions reduced in two directions are ambiguous, so that's hard...

# 0D boundary conditions --- easy case
@inline getbc(condition::XYZReducedField, ::Integer, ::Integer, ::AbstractGrid, args...) = @inbounds condition[1, 1, 1]

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

const XReducedAbstractField = AbstractField{Nothing}
const YReducedAbstractField = AbstractField{<:Any, Nothing}
const ZReducedAbstractField = AbstractField{<:Any, <:Any, Nothing}

const YZReducedAbstractField = AbstractField{<:Any, Nothing, Nothing}
const XZReducedAbstractField = AbstractField{Nothing, <:Any, Nothing}
const XYReducedAbstractField = AbstractField{Nothing, Nothing, <:Any}

const XYZReducedAbstractField = AbstractField{Nothing, Nothing, Nothing}

const ReducedAbstractField = Union{XReducedAbstractField,
                                   YReducedAbstractField,
                                   ZReducedAbstractField,
                                   YZReducedAbstractField,
                                   XZReducedAbstractField,
                                   XYReducedAbstractField,
                                   XYZReducedAbstractField}

# TODO: needs test
function LinearAlgebra.dot(a::AbstractField, b::AbstractField; condition=nothing)
    ca = condition_operand(a, condition, 0)
    cb = condition_operand(b, condition, 0)

    B = ca * cb # Binary operation
    r = zeros(a.grid, 1)

    Base.mapreducedim!(identity, +, r, B)
    return @allowscalar r[1]
end

function LinearAlgebra.norm(a::AbstractField; condition = nothing)
    r = zeros(a.grid, 1)
    Base.mapreducedim!(x -> x * x, +, r, condition_operand(a, condition, 0))
    return @allowscalar sqrt(r[1])
end

# TODO: in-place allocations with function mappings need to be fixed in Julia Base...
const SumReduction     = typeof(Base.sum!)
const MeanReduction    = typeof(Statistics.mean!)
const ProdReduction    = typeof(Base.prod!)
const MaximumReduction = typeof(Base.maximum!)
const MinimumReduction = typeof(Base.minimum!)
const AllReduction     = typeof(Base.all!)
const AnyReduction     = typeof(Base.any!)

initialize_reduced_field!(::SumReduction,     f, r::ReducedAbstractField, c) = Base.initarray!(interior(r), f, Base.add_sum, true, interior(c))
initialize_reduced_field!(::ProdReduction,    f, r::ReducedAbstractField, c) = Base.initarray!(interior(r), f, Base.mul_prod, true, interior(c))
initialize_reduced_field!(::AllReduction,     f, r::ReducedAbstractField, c) = Base.initarray!(interior(r), f, &, true, interior(c))
initialize_reduced_field!(::AnyReduction,     f, r::ReducedAbstractField, c) = Base.initarray!(interior(r), f, |, true, interior(c))
initialize_reduced_field!(::MaximumReduction, f, r::ReducedAbstractField, c) = Base.mapfirst!(f, interior(r), interior(c))
initialize_reduced_field!(::MinimumReduction, f, r::ReducedAbstractField, c) = Base.mapfirst!(f, interior(r), interior(c))

filltype(f, c) = eltype(c)
filltype(::Union{AllReduction, AnyReduction}, grid) = Bool

const PossibleLocs = Union{<:Nothing, <:Face, <:Center}

function reduced_location(loc::Tuple; dims)
    if dims isa Colon
        return (Nothing, Nothing, Nothing)
    else
        return Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)
    end
end

function reduced_location(loc::Tuple{<:PossibleLocs, <:PossibleLocs, <:PossibleLocs}; dims)
    if dims isa Colon
        return (nothing, nothing, nothing)
    else
        return Tuple(i ∈ dims ? nothing : loc[i] for i in 1:3)
    end
end

function reduced_dimension(loc)
    dims = ()
    for i in 1:3
        loc[i] == Nothing ? dims = (dims..., i) : dims
    end
    return dims
end

get_neutral_mask(::Union{AllReduction, AnyReduction})  = true
get_neutral_mask(::Union{SumReduction, MeanReduction}) = 0
get_neutral_mask(::ProdReduction)    = 1

# TODO make this Float32 friendly
get_neutral_mask(::MinimumReduction) = +Inf
get_neutral_mask(::MaximumReduction) = -Inf

"""
    condition_operand(f::Function, op::AbstractField, condition, mask)

Wrap `f(op)` in `ConditionedOperand` with `condition` and `mask`. `f` defaults to `identity`.

If `f isa identity` and `isnothing(condition)` then `op` is returned without wrapping.

Otherwise return `ConditionedOperand`, even when `isnothing(condition)` but `!(f isa identity)`.
"""
@inline condition_operand(op::AbstractField, condition, mask) = condition_operand(nothing, op, condition, mask)

# Do NOT condition if condition=nothing.
# All non-trivial conditioning is found in AbstractOperations/conditional_operations.jl
@inline condition_operand(::Nothing, operand, ::Nothing, mask) = operand

@inline conditional_length(c::AbstractField)        = length(c)
@inline conditional_length(c::AbstractField, dims)  = mapreduce(i -> size(c, i), *, unique(dims); init=1)

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin

        # In-place
        function Base.$(reduction!)(f::Function,
                                    r::ReducedAbstractField,
                                    a::AbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            operand = condition_operand(f, a, condition, mask)

            return Base.$(reduction!)(identity,
                                      interior(r),
                                      operand;
                                      kwargs...)
        end

        function Base.$(reduction!)(r::ReducedAbstractField,
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

            conditioned_c = condition_operand(f, c, condition, mask)
            T = filltype(Base.$(reduction!), c)
            loc = reduced_location(instantiated_location(c); dims)
            r = Field(loc, c.grid, T; indices=indices(c))
            initialize_reduced_field!(Base.$(reduction!), identity, r, conditioned_c)
            Base.$(reduction!)(identity, interior(r), conditioned_c, init=false)

            if dims isa Colon
                return @allowscalar first(r)
            else
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; kwargs...) = Base.$(reduction)(identity, c; kwargs...)
    end
end

# Improve me! We can should both the extrema in one single reduction instead of two
Base.extrema(c::AbstractField; kwargs...) = (minimum(c; kwargs...), maximum(c; kwargs...))
Base.extrema(f, c::AbstractField; kwargs...) = (minimum(f, c; kwargs...), maximum(f, c; kwargs...))

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

function Statistics.mean!(f::Function, r::ReducedAbstractField, a::AbstractField; condition = nothing, mask = 0)
    sum!(f, r, a; condition, mask, init=true)
    dims = reduced_dimension(location(r))
    n = conditional_length(condition_operand(f, a, condition, mask), dims)
    r ./= n
    return r
end

Statistics.mean!(r::ReducedAbstractField, a::AbstractArray; kwargs...) = Statistics.mean!(identity, r, a; kwargs...)

function Base.isapprox(a::AbstractField, b::AbstractField; kw...)
    conditional_a = condition_operand(a, nothing, one(eltype(a)))
    conditional_b = condition_operand(b, nothing, one(eltype(b)))
    # TODO: Make this non-allocating?
    return all(isapprox.(conditional_a, conditional_b; kw...))
end

#####
##### fill_halo_regions!
#####

function fill_halo_regions!(field::Field, positional_args...; kwargs...) 

    arch = architecture(field.grid)
    args = (field.data,
            field.boundary_conditions,
            field.indices,
            instantiated_location(field),
            field.grid,
            positional_args...)
    
    # Manually convert args... to be 
    # passed to the fill_halo_regions! function.
    GC.@preserve args begin
        converted_args = convert_to_device(arch, args)
        fill_halo_regions!(converted_args...; kwargs...)
    end

    return nothing
end

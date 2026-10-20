using GPUArraysCore: @allowscalar
using Oceananigans.AbstractOperations: AbstractOperations, Averaging, conditional_one, validate_condition
using Oceananigans.Fields: Fields, AbstractField, ReducedAbstractField, Scan, AbstractReducing, AbstractAccumulating,
    AllReduction, AnyReduction, MaximumReduction, MinimumReduction, ProdReduction, SumReduction,
    condition_operand, conditional_length, filltype, get_neutral_mask, indices,
    initialize_reduced_field!, reduced_dimensions, reduced_location, reduction_operand, scan_indices
using Oceananigans.ImmersedBoundaries: NotImmersed, NotImmersedColumn
using Oceananigans.Utils: tupleit
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra, dot

# A reduction whose `dims` include every partitioned dimension "spans the partition": the regional results
# are combined into a field on the first region. Other reductions are computed region by region.

const MultiRegionAbstractField = AbstractField{<:Any, <:Any, <:Any, <:MultiRegionGrids}

partitioned_dimensions(::XPartition) = tuple(1)
partitioned_dimensions(::YPartition) = tuple(2)
partitioned_dimensions(::AbstractPartition) = (1, 2)

function spans_partition(grid::MultiRegionGrids, dims)
    reduced_dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    partitioned = partitioned_dimensions(grid.partition)
    spanned = map(d -> d ∈ reduced_dims, partitioned)
    all(spanned) || !any(spanned) ||
        throw(ArgumentError("Reductions over dims=$reduced_dims are not supported on a MultiRegionGrid partitioned along dimensions $partitioned."))
    return all(spanned)
end

reduction_grid(grid::MultiRegionGrids, dims) = spans_partition(grid, dims) ? getregion(grid, 1) : grid

Fields.scan_grid(::AbstractReducing, grid::MultiRegionGrids, dims) = reduction_grid(grid, dims)

function Fields.scan_grid(::AbstractAccumulating, grid::MultiRegionGrids, dims)
    spans_partition(grid, dims) &&
        throw(ArgumentError("Accumulations over dims=$dims are not supported on a MultiRegionGrid partitioned along dimensions $(partitioned_dimensions(grid.partition))."))
    return grid
end

Fields.scan_indices(type::AbstractReducing, indices::MultiRegionObject, dims) = scan_indices(type, getregion(indices, 1), dims)
Fields.scan_indices(type::AbstractAccumulating, indices::MultiRegionObject, dims) = scan_indices(type, getregion(indices, 1), dims)

#####
##### Conditions are evaluated region by region
#####

# A global array condition is split along the partitioned dimension into regional arrays
regional_condition(condition, operand) = condition
regional_condition(condition::MultiRegionAbstractField, operand) = condition
regional_condition(condition::AbstractArray, operand) = partition_condition(condition, operand, operand.grid.partition)

partition_condition(condition, operand, ::XPartition) = split_condition(condition, operand, 1)
partition_condition(condition, operand, ::YPartition) = split_condition(condition, operand, 2)
partition_condition(condition, operand, partition) =
    throw(ArgumentError("A global array cannot be a condition on a MultiRegionGrid with $(summary(partition)): use a MultiRegionObject of regional arrays."))

function split_condition(condition, operand, dim)
    regional_sizes = Tuple(size(getregion(operand, r), dim) for r in regions(operand.grid))
    last_indices = cumsum(regional_sizes)
    first_indices = last_indices .- regional_sizes .+ 1

    last(last_indices) == size(condition, dim) ||
        throw(ArgumentError("The condition has size $(size(condition)), but the regions add up to $(last(last_indices)) points along dimension $dim."))

    return MultiRegionObject(map((i, j) -> copy(selectdim(condition, dim, i:j)), first_indices, last_indices))
end

function validate_regional_condition(condition, operand)
    foreach(r -> validate_condition(getregion(condition, r), getregion(operand, r)), regions(operand.grid))
    return condition
end

AbstractOperations.validate_condition(condition::AbstractArray, operand::MultiRegionAbstractField) =
    validate_regional_condition(regional_condition(condition, operand), operand)

AbstractOperations.validate_condition(condition::MultiRegionObject, operand::MultiRegionAbstractField) =
    validate_regional_condition(condition, operand)

# Immersed conditions wrap the user condition, which may be regional
@inline Utils.getregion(ni::NotImmersed, r)  = NotImmersed(_getregion(ni.condition, r))
@inline Utils._getregion(ni::NotImmersed, r) = NotImmersed(getregion(ni.condition, r))

@inline Utils.getregion(nic::NotImmersedColumn, r)  = NotImmersedColumn(_getregion(nic.immersed_column, r), _getregion(nic.condition, r))
@inline Utils._getregion(nic::NotImmersedColumn, r) = NotImmersedColumn(getregion(nic.immersed_column, r), getregion(nic.condition, r))

#####
##### Reductions
#####

# The points of overlapping regions that a reduction spanning the partition counts
spanning_restriction(reduction!, grid) = nothing
restrict_operand(operand, ::Nothing, mask) = operand

# Reduce one region of `a` into the single-region field `r`
function reduce_region!(reduction!, f, r, a, condition, mask, init, restriction=nothing)
    operand = restrict_operand(condition_operand(f, a, condition, mask), restriction, mask)
    init && initialize_reduced_field!(reduction!, identity, r, operand)
    reduction!(identity, interior(r), reduction_operand(operand); init=false)
    return r
end

# A reduction spanning the partition reduces each region into a partial result that is combined with `r`:
# GPU reductions with `init=false` use the output as the neutral element of every thread block.
function reduce_regions!(reduction!, f, r, a, condition, mask, init)
    dims = reduced_dimensions(r)
    spans_partition(a.grid, dims) ||
        throw(ArgumentError("The result of a reduction over dims=$dims must be a reduced field on the MultiRegionGrid."))

    restriction = spanning_restriction(reduction!, a.grid)
    partial = Field(instantiated_location(r), r.grid, eltype(r); indices=indices(r))
    combine = reduction_operator(reduction!)

    for region in regions(a.grid)
        reduce_region!(reduction!, f, partial, getregion(a, region), getregion(condition, region), mask, true, restriction)
        if init && region == first(regions(a.grid))
            interior(r) .= interior(partial)
        else
            interior(r) .= combine.(interior(r), interior(partial))
        end
    end

    return r
end

reduction_operator(::SumReduction)     = +
reduction_operator(::ProdReduction)    = *
reduction_operator(::MaximumReduction) = max
reduction_operator(::MinimumReduction) = min
reduction_operator(::AllReduction)     = &
reduction_operator(::AnyReduction)     = |

# A reduction that spans the partition is combined in the first region and copied to the others
function reduce_regions!(reduction!, f, r::MultiRegionAbstractField, a, condition, mask, init)
    if spans_partition(a.grid, reduced_dimensions(r))
        reduce_regions!(reduction!, f, getregion(r, 1), a, condition, mask, init)
        for region in regions(r.grid)[2:end]
            interior(getregion(r, region)) .= interior(getregion(r, 1))
        end
    else
        for region in regions(a.grid)
            reduce_region!(reduction!, f, getregion(r, region), getregion(a, region), getregion(condition, region), mask, init)
        end
    end
    return r
end

function reduced_multi_region_field(T, c, dims)
    loc = reduced_location(instantiated_location(c); dims)
    return Field(loc, reduction_grid(c.grid, dims), T; indices=getregion(indices(c), 1))
end

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin
        function Base.$(reduction!)(f::Function, r::ReducedAbstractField, a::MultiRegionAbstractField;
                                    condition = nothing, mask = get_neutral_mask(Base.$(reduction!)), init = true)
            mask = convert(filltype(Base.$(reduction!), a), mask)
            return reduce_regions!(Base.$(reduction!), f, r, a, regional_condition(condition, a), mask, init)
        end

        Base.$(reduction!)(r::ReducedAbstractField, a::MultiRegionAbstractField; kwargs...) =
            Base.$(reduction!)(identity, r, a; kwargs...)

        function Base.$(reduction)(f::Function, c::MultiRegionAbstractField;
                                   condition = nothing, mask = get_neutral_mask(Base.$(reduction!)), dims = :)
            r = reduced_multi_region_field(filltype(Base.$(reduction!), c), c, dims)
            Base.$(reduction!)(f, r, c; condition, mask)
            return reduction_result(r, dims)
        end
    end
end

reduction_result(r, ::Colon) = @allowscalar interior(r)[1, 1, 1]
reduction_result(r, dims) = r

#####
##### Means divide the sum by the number of points that satisfy the condition
#####

function multi_region_length(c, grid, dims)
    regional_lengths = Tuple(conditional_length(getregion(c, r), dims) for r in regions(grid))
    return spans_partition(grid, dims) ? sum(regional_lengths) : first(regional_lengths)
end

# Conditional operations, including those that exclude immersed points, count with `sum(conditional_one(...))`;
# their methods resolve the ambiguity with the field methods.
const MultiRegionGridField = AbstractField{<:Any, <:Any, <:Any, <:MultiRegionGrid}
const MultiRegionGridConditionalOperation = ConditionalOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:MultiRegionGrid}

Fields.conditional_length(c::MultiRegionGridField) = multi_region_length(c, c.grid, :)
Fields.conditional_length(c::MultiRegionGridField, dims::Int) = multi_region_length(c, c.grid, dims)
Fields.conditional_length(c::MultiRegionGridField, dims::Tuple) = multi_region_length(c, c.grid, dims)

Fields.conditional_length(c::MultiRegionGridConditionalOperation) = sum(conditional_one(c, 0))
Fields.conditional_length(c::MultiRegionGridConditionalOperation, dims::Int) = sum(conditional_one(c, 0); dims)
Fields.conditional_length(c::MultiRegionGridConditionalOperation, dims::Tuple) = sum(conditional_one(c, 0); dims)

divide_by_length!(total::Number, L) = total / L

function divide_by_length!(r, L)
    @apply_regionally regional_divide!(r, L)
    return r
end

regional_divide!(r, L::Number) = interior(r) ./= L
regional_divide!(r, L::AbstractField) = interior(r) ./= interior(L)

function multi_region_mean(f, c, dims; condition = nothing, mask = 0)
    mask = convert(eltype(c), mask)
    condition = regional_condition(condition, c)
    total = sum(f, c; condition, mask, dims)
    L = conditional_length(condition_operand(f, c, condition, mask), dims)
    return divide_by_length!(total, L)
end

Statistics._mean(f, c::MultiRegionAbstractField, dims::Colon; kwargs...) = multi_region_mean(f, c, dims; kwargs...)
Statistics._mean(f, c::MultiRegionAbstractField, dims; kwargs...) = multi_region_mean(f, c, dims; kwargs...)

function Statistics.mean!(f::Function, r::ReducedAbstractField, a::MultiRegionAbstractField; condition = nothing, mask = 0)
    mask = convert(eltype(a), mask)
    condition = regional_condition(condition, a)
    sum!(f, r, a; condition, mask)
    L = conditional_length(condition_operand(f, a, condition, mask), reduced_dimensions(r))
    return divide_by_length!(r, L)
end

#####
##### Scans (`Reduction`, `Integral`, `Average`, ...) of operands on multi-region grids
#####

@inline Utils.getregion(s::Scan, r)  = Scan(_getregion(s.type, r), s.scan!, _getregion(s.operand, r), s.dims)
@inline Utils._getregion(s::Scan, r) = Scan(getregion(s.type, r), s.scan!, getregion(s.operand, r), s.dims)

@inline Utils.getregion(a::Averaging, r)  = Averaging(_getregion(a.volume, r))
@inline Utils._getregion(a::Averaging, r) = Averaging(getregion(a.volume, r))

#####
##### Dot products
#####

const MRD = Union{MultiRegionField, MultiRegionObject}

# make it more efficient?
LinearAlgebra.dot(f::MRD, g::MRD; condition = nothing) =
    sum(construct_regionally(dot, f, g; condition).regional_objects)
LinearAlgebra.norm(f::MRD) = sqrt(dot(f, f))

# The regional dot products are summed on the host; all regions share one device and hence one `r`
Fields.dot!(r, f::MRD, g::MRD; condition = nothing) = fill!(r, dot(f, g; condition))

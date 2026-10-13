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

#####
##### Reductions of fields on MultiRegionGrids
#####
##### A reduction whose `dims` include every partitioned dimension "spans the partition": the regional results
##### are combined into a field on the first region. Other reductions are computed region by region.
#####

const MultiRegionAbstractField = AbstractField{<:Any, <:Any, <:Any, <:MultiRegionGrids}

partitioned_dimensions(::XPartition) = tuple(1)
partitioned_dimensions(::YPartition) = tuple(2)
partitioned_dimensions(::AbstractPartition) = (1, 2)

normalize_dims(::Colon) = (1, 2, 3)
normalize_dims(dims) = tupleit(dims)

function spans_partition(grid::MultiRegionGrids, dims)
    reduced_dims = normalize_dims(dims)
    partitioned = partitioned_dimensions(grid.partition)
    spanned = map(d -> d ∈ reduced_dims, partitioned)

    all(spanned) && return true

    if any(spanned)
        throw(ArgumentError("Reductions over dims=$reduced_dims are not supported on a MultiRegionGrid partitioned along dimensions $partitioned."))
    end

    return false
end

reduction_grid(grid::MultiRegionGrids, dims) = spans_partition(grid, dims) ? getregion(grid, 1) : grid

function accumulation_grid(grid::MultiRegionGrids, dims)
    spans_partition(grid, dims) &&
        throw(ArgumentError("Accumulations over dims=$dims are not supported on a MultiRegionGrid partitioned along dimensions $(partitioned_dimensions(grid.partition))."))

    return grid
end

Fields.scan_grid(::AbstractReducing, grid::MultiRegionGrids, dims) = reduction_grid(grid, dims)
Fields.scan_grid(::AbstractAccumulating, grid::MultiRegionGrids, dims) = accumulation_grid(grid, dims)

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

    regional_conditions = map((i, j) -> copy(selectdim(condition, dim, i:j)), first_indices, last_indices)

    return MultiRegionObject(regional_conditions)
end

function validate_regional_condition(condition, operand)
    for r in regions(operand.grid)
        validate_condition(getregion(condition, r), getregion(operand, r))
    end
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
    operand = condition_operand(f, a, condition, mask)
    operand = restrict_operand(operand, restriction, mask)
    init && initialize_reduced_field!(reduction!, identity, r, operand)
    reduction!(identity, interior(r), reduction_operand(operand); init=false)
    return r
end

# The other regions are reduced into a partial result and then combined with `r`: GPU reductions with
# `init=false` use the output as the neutral element of every thread block.
function reduce_regions!(reduction!, f, r, a, condition, mask, init)
    dims = reduced_dimensions(r)

    spans_partition(a.grid, dims) ||
        throw(ArgumentError("The result of a reduction over dims=$dims must be a reduced field on the MultiRegionGrid."))

    restriction = spanning_restriction(reduction!, a.grid)
    first_region = first(regions(a.grid))
    other_regions = regions(a.grid)[2:end]

    reduce_region!(reduction!, f, r, getregion(a, first_region), getregion(condition, first_region), mask, init, restriction)

    isempty(other_regions) && return r

    partial = Field(instantiated_location(r), r.grid, eltype(r); indices=indices(r))

    for region in other_regions
        reduce_region!(reduction!, f, partial, getregion(a, region), getregion(condition, region), mask, true, restriction)
        combine_partial_reductions!(reduction!, interior(r), interior(partial))
    end

    return r
end

combine_partial_reductions!(reduction!, R, P) = R .= reduction_operator(reduction!).(R, P)

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
        copy_first_region!(r)
    else
        for region in regions(a.grid)
            reduce_region!(reduction!, f, getregion(r, region), getregion(a, region), getregion(condition, region), mask, init)
        end
    end

    return r
end

function copy_first_region!(r)
    first_interior = interior(getregion(r, 1))

    for region in regions(r.grid)[2:end]
        interior(getregion(r, region)) .= first_interior
    end

    return r
end

function reduce_multi_region!(reduction!, f, r, a, condition, mask, init)
    typed_mask = convert(filltype(reduction!, a), mask)
    regional_conditions = regional_condition(condition, a)
    return reduce_regions!(reduction!, f, r, a, regional_conditions, typed_mask, init)
end

function reduced_multi_region_field(T, c, dims)
    loc = reduced_location(instantiated_location(c); dims)
    grid = reduction_grid(c.grid, dims)
    regional_indices = getregion(indices(c), 1)
    return Field(loc, grid, T; indices=regional_indices)
end

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin
        # In-place
        function Base.$(reduction!)(f::Function,
                                    r::ReducedAbstractField,
                                    a::MultiRegionAbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    init = true)

            return reduce_multi_region!(Base.$(reduction!), f, r, a, condition, mask, init)
        end

        Base.$(reduction!)(r::ReducedAbstractField, a::MultiRegionAbstractField; kwargs...) =
            Base.$(reduction!)(identity, r, a; kwargs...)

        # Allocating
        function Base.$(reduction)(f::Function,
                                   c::MultiRegionAbstractField;
                                   condition = nothing,
                                   mask = get_neutral_mask(Base.$(reduction!)),
                                   dims = :)

            T = filltype(Base.$(reduction!), c)
            r = reduced_multi_region_field(T, c, dims)
            reduce_multi_region!(Base.$(reduction!), f, r, c, condition, mask, true)

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

divide_by_length!(r, L::Number) = interior(r) ./= L
divide_by_length!(r, L::AbstractField) = interior(r) ./= interior(L)

function Statistics._mean(f, c::MultiRegionAbstractField, ::Colon; condition = nothing, mask = 0)
    typed_mask = convert(eltype(c), mask)
    regional_conditions = regional_condition(condition, c)
    total = sum(f, c; condition = regional_conditions, mask = typed_mask)
    L = conditional_length(condition_operand(f, c, regional_conditions, typed_mask))
    return total / L
end

function Statistics._mean(f, c::MultiRegionAbstractField, dims; condition = nothing, mask = 0)
    typed_mask = convert(eltype(c), mask)
    regional_conditions = regional_condition(condition, c)
    r = sum(f, c; condition = regional_conditions, mask = typed_mask, dims)
    L = conditional_length(condition_operand(f, c, regional_conditions, typed_mask), dims)
    @apply_regionally divide_by_length!(r, L)
    return r
end

function Statistics.mean!(f::Function, r::ReducedAbstractField, a::MultiRegionAbstractField; condition = nothing, mask = 0)
    typed_mask = convert(eltype(a), mask)
    regional_conditions = regional_condition(condition, a)
    sum!(f, r, a; condition = regional_conditions, mask = typed_mask)
    L = conditional_length(condition_operand(f, a, regional_conditions, typed_mask), reduced_dimensions(r))
    @apply_regionally divide_by_length!(r, L)
    return r
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

using GPUArraysCore: @allowscalar
using Oceananigans.AbstractOperations: Averaging
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

Fields.scan_grid(::AbstractAccumulating, grid::MultiRegionGrids, dims) = spans_partition(grid, dims) ?
    throw(ArgumentError("Accumulations over dims=$dims are not supported on a MultiRegionGrid partitioned along dimensions $(partitioned_dimensions(grid.partition)).")) :
    grid

Fields.scan_indices(type::AbstractReducing, indices::MultiRegionObject, dims) = scan_indices(type, getregion(indices, 1), dims)
Fields.scan_indices(type::AbstractAccumulating, indices::MultiRegionObject, dims) = scan_indices(type, getregion(indices, 1), dims)

# Scans and immersed conditions wrap operands and conditions that may be regional
@inline Utils.getregion(s::Scan, r)  = Scan(_getregion(s.type, r), s.scan!, _getregion(s.operand, r), s.dims)
@inline Utils._getregion(s::Scan, r) = Scan(getregion(s.type, r), s.scan!, getregion(s.operand, r), s.dims)

@inline Utils.getregion(a::Averaging, r)  = Averaging(_getregion(a.volume, r))
@inline Utils._getregion(a::Averaging, r) = Averaging(getregion(a.volume, r))

@inline Utils.getregion(ni::NotImmersed, r)  = NotImmersed(_getregion(ni.condition, r))
@inline Utils._getregion(ni::NotImmersed, r) = NotImmersed(getregion(ni.condition, r))

@inline Utils.getregion(nic::NotImmersedColumn, r)  = NotImmersedColumn(_getregion(nic.immersed_column, r), _getregion(nic.condition, r))
@inline Utils._getregion(nic::NotImmersedColumn, r) = NotImmersedColumn(getregion(nic.immersed_column, r), getregion(nic.condition, r))

#####
##### Reductions
#####

# The operand of a reduction spanning the partition, restricted on Yin-Yang grids to the points each panel owns
spanning_operand(reduction!, operand) = operand

reduction_operator(::SumReduction)     = +
reduction_operator(::ProdReduction)    = *
reduction_operator(::MaximumReduction) = max
reduction_operator(::MinimumReduction) = min
reduction_operator(::AllReduction)     = &
reduction_operator(::AnyReduction)     = |

function reduce_region!(reduction!, r, operand)
    initialize_reduced_field!(reduction!, identity, r, operand)
    reduction!(identity, interior(r), reduction_operand(operand); init=false)
    return r
end

# A reduction spanning the partition combines the result of the first region with the partial results of the
# others: GPU reductions with `init=false` use the output as the neutral element of every thread block.
function reduce_regions!(reduction!, r, operand)
    dims = reduced_dimensions(r)
    spans_partition(operand.grid, dims) ||
        throw(ArgumentError("The result of a reduction over dims=$dims must be a reduced field on the MultiRegionGrid."))

    operand = spanning_operand(reduction!, operand)
    partial = Field(instantiated_location(r), r.grid, eltype(r); indices=indices(r))
    combine = reduction_operator(reduction!)
    reduce_region!(reduction!, r, getregion(operand, 1))

    for region in regions(operand.grid)[2:end]
        reduce_region!(reduction!, partial, getregion(operand, region))
        interior(r) .= combine.(interior(r), interior(partial))
    end

    return r
end

function reduce_regions!(reduction!, r::MultiRegionAbstractField, operand)
    for region in regions(operand.grid)
        reduce_region!(reduction!, getregion(r, region), getregion(operand, region))
    end
    return r
end

reduced_multi_region_field(T, c, dims) =
    Field(reduced_location(instantiated_location(c); dims), reduction_grid(c.grid, dims), T; indices=getregion(indices(c), 1))

reduction_result(r, ::Colon) = @allowscalar interior(r)[1, 1, 1]
reduction_result(r, dims) = r

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin
        function Base.$(reduction!)(f::Function, r::ReducedAbstractField, a::MultiRegionAbstractField;
                                    condition = nothing, mask = get_neutral_mask(Base.$(reduction!)))
            mask = convert(filltype(Base.$(reduction!), a), mask)
            return reduce_regions!(Base.$(reduction!), r, condition_operand(f, a, condition, mask))
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

#####
##### Means divide the sum by the number of points that satisfy the condition, counted by the same reduction
#####

# Points are counted through a condition, which on Yin-Yang grids weights them by their owned fractions
counted_operand(operand) = ConditionalOperation(operand)
counted_operand(operand::ConditionalOperation) = operand

regional_divide!(r, L) = interior(r) ./= interior(L)

function Statistics.mean!(f::Function, r::ReducedAbstractField, a::MultiRegionAbstractField; condition = nothing, mask = 0)
    mask = convert(eltype(a), mask)
    sum!(f, r, a; condition, mask)
    L = conditional_length(counted_operand(condition_operand(f, a, condition, mask)), reduced_dimensions(r))
    @apply_regionally regional_divide!(r, L)
    return r
end

function multi_region_mean(f, c, dims; kwargs...)
    r = reduced_multi_region_field(eltype(c), c, dims)
    Statistics.mean!(f, r, c; kwargs...)
    return reduction_result(r, dims)
end

Statistics._mean(f, c::MultiRegionAbstractField, dims::Colon; kwargs...) = multi_region_mean(f, c, dims; kwargs...)
Statistics._mean(f, c::MultiRegionAbstractField, dims; kwargs...) = multi_region_mean(f, c, dims; kwargs...)

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

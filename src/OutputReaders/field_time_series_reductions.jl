using Statistics

using Oceananigans: instantiated_location
using Oceananigans.Fields: reduced_location

import Oceananigans.Fields: conditional_length

@inline conditional_length(fts::FieldTimeSeries) = length(fts) * conditional_length(fts[1])

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::AbstractFieldTimeSeries) = (size(fts.grid, location(fts), indices(fts))..., length(fts.times))
@propagate_inbounds Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

#####
##### Reductions, shared by FieldTimeSeries and FieldTimeSeriesOperation
#####

# The slice at time index n in a form that lazy reductions (`sum(f, slice)`,
# `sum!(f, dest, slice)`) accept: a Field for stored series, a lazy operation for
# operations (reductions apply pointwise without materializing them).
reduction_time_slice(fts::FieldTimeSeries, n) = fts[n]

# The slice at time index n as an interior-accessible Field, accumulated into the
# reusable buffer `temp` where one is needed (see materialized_time_slice, which
# produces the first slice and the buffer).
materialized_time_slice!(temp, fts::FieldTimeSeries, n) = fts[n]

# Element-wise accumulation operations for time dimension reduction,
# applying the mapped function `f` to the accumulated values
@inline broadcasted_accumulate!(::typeof(sum!), r, f, a) = r .+= f.(a)
@inline broadcasted_accumulate!(::typeof(prod!), r, f, a) = r .*= f.(a)
@inline broadcasted_accumulate!(::typeof(maximum!), r, f, a) = r .= max.(r, f.(a))
@inline broadcasted_accumulate!(::typeof(minimum!), r, f, a) = r .= min.(r, f.(a))
@inline broadcasted_accumulate!(::typeof(all!), r, f, a) = r .&= f.(a)
@inline broadcasted_accumulate!(::typeof(any!), r, f, a) = r .|= f.(a)

tupleit(a) = tuple(a)
tupleit(a::Tuple) = a
tupleit(a::Colon) = Colon()

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin

        # Allocating
        function Base.$(reduction)(f::Function, fts::AbstractFieldTimeSeries; dims=:, kw...)
            dims = tupleit(dims)

            if dims isa Colon
                return Base.$(reduction)($(reduction)(f, reduction_time_slice(fts, n); kw...) for n in 1:length(fts.times))
            elseif 4 ∈ dims
                # Reduce over time dimension
                spatial_dims = filter(d -> d != 4, dims)
                loc = isempty(spatial_dims) ? instantiated_location(fts) : reduced_location(instantiated_location(fts); dims=spatial_dims)
                new_times = [mean(fts.times)]
                rts = FieldTimeSeries(loc, fts.grid, new_times; indices=indices(fts), kw...)
                return Base.$(reduction!)(f, rts, fts; dims, kw...)
            else
                loc = reduced_location(instantiated_location(fts); dims)
                times = fts.times
                rts = FieldTimeSeries(loc, fts.grid, times; indices=indices(fts), kw...)
                return Base.$(reduction!)(f, rts, fts; dims, kw...)
            end
        end

        Base.$(reduction)(fts::AbstractFieldTimeSeries; kw...) = Base.$(reduction)(identity, fts; kw...)

        function Base.$(reduction!)(f::Function, rts::FieldTimeSeries, fts::AbstractFieldTimeSeries; dims=:, kw...)
            dims = tupleit(dims)
            Nt = length(fts.times)

            if 4 ∈ dims
                # Reduce over time dimension
                spatial_dims = filter(d -> d != 4, dims)
                if isempty(spatial_dims)
                    # Initialize with the f-mapped first slice, then accumulate
                    temp = materialized_time_slice(fts, 1)
                    interior(rts[1]) .= f.(interior(temp))
                    for n = 2:Nt
                        temp = materialized_time_slice!(temp, fts, n)
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), f, interior(temp))
                    end
                else
                    # Use the allocating reduction for the first step, then reuse temp
                    temp = Base.$(reduction)(f, reduction_time_slice(fts, 1); dims=spatial_dims, kw...)
                    set!(rts[1], temp)
                    for n = 2:Nt
                        Base.$(reduction!)(f, temp, reduction_time_slice(fts, n); kw...)
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), identity, interior(temp))
                    end
                end
            else
                # For spatial-only reductions, the result field already has the correct location
                for n = 1:Nt
                    Base.$(reduction!)(f, rts[n], reduction_time_slice(fts, n); kw...)
                end
            end
            return rts
        end

        Base.$(reduction!)(rts::FieldTimeSeries, fts::AbstractFieldTimeSeries; kw...) = Base.$(reduction!)(identity, rts, fts; kw...)
    end
end

function Statistics._mean(f, c::AbstractFieldTimeSeries, ::Colon; condition=nothing)
    condition == nothing || error("condition_operand for FieldTimeSeries not implemented")
    # TODO: implement condition_operand for FieldTimeSeries?
    # operator = condition_operand(f, c, condition, mask)
    return sum(f, c) / conditional_length(c)
end

function Statistics._mean(f, c::AbstractFieldTimeSeries, dims; condition=nothing)
    condition == nothing || error("condition_operand for FieldTimeSeries not implemented")
    # TODO: implement condition_operand for FieldTimeSeries?
    # operand = condition_operand(f, c, condition, mask)
    r = sum(f, c; dims)
    n = conditional_length(c, dims)
    r ./= n
    return r
end

using Statistics

using Oceananigans: instantiated_location
using Oceananigans.Fields: reduced_location, filltype

import Oceananigans.Fields: conditional_length

@inline conditional_length(fts::FieldTimeSeries) = length(fts) * conditional_length(fts[1])

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(fts.grid, location(fts), fts.indices)..., length(fts.times))
@propagate_inbounds Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

#####
##### Basic support for reductions
#####

# Element-wise accumulation operations for time dimension reduction
@inline broadcasted_accumulate!(::typeof(sum!), r, a) = r .+= a
@inline broadcasted_accumulate!(::typeof(prod!), r, a) = r .*= a
@inline broadcasted_accumulate!(::typeof(maximum!), r, a) = r .= max.(r, a)
@inline broadcasted_accumulate!(::typeof(minimum!), r, a) = r .= min.(r, a)
@inline broadcasted_accumulate!(::typeof(all!), r, a) = r .&= a
@inline broadcasted_accumulate!(::typeof(any!), r, a) = r .|= a

tupleit(a) = tuple(a)
tupleit(a::Tuple) = a
tupleit(a::Colon) = Colon()

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin

        # Allocating
        function Base.$(reduction)(f::Function, fts::FTS; dims=:, kw...)
            dims = tupleit(dims)

            if dims isa Colon
                return Base.$(reduction)($(reduction)(f, fts[n]; kw...) for n in 1:length(fts.times))
            elseif 4 ∈ dims
                # Reduce over time dimension
                spatial_dims = filter(d -> d != 4, dims)
                loc = isempty(spatial_dims) ? instantiated_location(fts) : reduced_location(instantiated_location(fts); dims=spatial_dims)
                new_times = [mean(fts.times)]
                rts = FieldTimeSeries(loc, fts.grid, new_times; indices=fts.indices, kw...)
                return Base.$(reduction!)(f, rts, fts; dims, kw...)
            else
                loc = reduced_location(instantiated_location(fts); dims)
                times = fts.times
                rts = FieldTimeSeries(loc, fts.grid, times; indices=fts.indices, kw...)
                return Base.$(reduction!)(f, rts, fts; dims, kw...)
            end
        end

        Base.$(reduction)(fts::FTS; kw...) = Base.$(reduction)(identity, fts; kw...)

        function Base.$(reduction!)(f::Function, rts::FTS, fts::FTS; dims=:, kw...)
            dims = tupleit(dims)

            if 4 ∈ dims
                # Reduce over time dimension
                spatial_dims = filter(d -> d != 4, dims)
                # Initialize with the first time step
                if isempty(spatial_dims)
                    set!(rts[1], fts[1])
                    for n = 2:length(fts)
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), interior(fts[n]))
                    end
                else
                    # Use the allocating reduction for the first step
                    temp = Base.$(reduction)(f, fts[1]; dims=spatial_dims, kw...)
                    set!(rts[1], temp)
                    for n = 2:length(fts)
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), interior(fts[n]))
                    end
                end
            else
                # For spatial-only reductions, the result field already has the correct location
                for n = 1:length(rts)
                    Base.$(reduction!)(f, rts[n], fts[n]; kw...)
                end
            end
            return rts
        end

        Base.$(reduction!)(rts::FTS, fts::FTS; kw...) = Base.$(reduction!)(identity, rts, fts; kw...)
    end
end

function Statistics._mean(f, c::FTS, ::Colon; condition=nothing)
    condition == nothing || error("condition_operand for FieldTimeSeries not implemented")
    # TODO: implement condition_operand for FieldTimeSeries?
    # operator = condition_operand(f, c, condition, mask)
    return sum(c) / conditional_length(c)
end

function Statistics._mean(f, c::FTS, dims; condition=nothing)
    condition == nothing || error("condition_operand for FieldTimeSeries not implemented")
    # TODO: implement condition_operand for FieldTimeSeries?
    # operand = condition_operand(f, c, condition, mask)
    r = sum(c; dims)
    n = conditional_length(c, dims)
    r ./= n
    return r
end

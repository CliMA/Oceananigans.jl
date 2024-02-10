using Statistics

#####
##### Reductions
#####

Statistics.mean(f::Function, fts::FieldTimeSeries; dims=:) =
    mean(f, parent(fts); dims)

function Statistics.mean(fts::FieldTimeSeries; dims=:)
    return mean(parent(fts), dims)
    #=
    m = mean(fts[1]; dims)
    Nt = length(fts)

    if dims isa Colon
        for n = 2:Nt
            m += mean(fts[n])
        end

        return m / Nt
    else
        for n = 2:Nt
            m .+= mean(fts[n]; dims)
        end

        m ./= Nt

        return m
    end
    =#
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(fts.grid, location(fts), fts.indices)..., length(fts.times))
@propagate_inbounds Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

#####
##### Basic support for reductions
#####
##### TODO: support for reductions across _time_ (ie when 4 ∈ dims)
#####

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin

        # Allocating
        function Base.$(reduction)(f::Function, fts::FTS; dims=:, kw...)
            if dims isa Colon        
                return Base.$(reduction)($(reduction)(f, fts[n]; kw...) for n in 1:length(fts.times))
            else
                T = filltype(Base.$(reduction!), fts)
                loc = LX, LY, LZ = reduced_location(location(fts); dims)
                times = fts.times
                rts = FieldTimeSeries{LX, LY, LZ}(grid, times, T; indices=fts.indices)
                return Base.$(reduction!)(f, rts, fts; kw...)
            end
        end

        Base.$(reduction)(fts::FTS; kw...) = Base.$(reduction)(identity, fts; kw...)

        function Base.$(reduction!)(f::Function,rts::FTS, fts::FTS; dims=:, kw...)
            dims isa Tuple && 4 ∈ dims && error("Reduction across the time dimension (dim=4) is not yet supported!")
            for n = 1:length(rts)
                Base.$(reduction!)(f, rts[i], fts[i]; dims, kw...)
            end
            return rts
        end

        Base.$(reduction!)(rts::FTS, fts::FTS; kw...) = Base.$(reduction!)(identity, rts, fts; kw...)
    end
end


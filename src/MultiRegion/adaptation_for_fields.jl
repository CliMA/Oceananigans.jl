
using Oceananigans.Fields: get_neutral_mask

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')
    @eval begin
        # Allocating
        function Base.$(reduction)(f::Function, mrf::MultiRegionField;
            condition = nothing, mask = get_neutral_mask(Base.$(reduction!)), dims=:) 
            args   = (f, fields(mrf))
            iter   = (0, 1)
            kwargs = (condition = condition, mask = mask, dims = dims)
            r = multi_region_object(mrf, Base.$(reduction), args, iter, kwargs) 
            return r
        end
    end
end
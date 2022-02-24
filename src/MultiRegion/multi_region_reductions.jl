using Statistics
import Statistics.mean
import Statistics.norm

reductions = (:(Base.sum), :(Base.maximum), :(Base.minimum), :(Base.prod), :(Base.any), :(Base.all), :(Statistics.mean))

# Allocating reductions
for reduction in reductions
    @eval begin
        function $(reduction)(f::Function, c::MultiRegionField; kwargs...)
            mr = construct_regionally($(reduction), f, c; kwargs...)
            if mr.regions isa NTuple{<:Any, <:Number}
                return $(reduction)([r for r in mr.regions]) 
            else
                FT   = eltype(mr.regions[1])
                loc  = location(mr.regions[1])
                validate_reduction_location!(loc, c.grid.partition)
                mrg  = MultiRegionGrid{FT, loc[1], loc[2], loc[3]}(architecture(c), c.grid.partition, MultiRegionObject(collect_grid(mr.regions), devices(mr)), devices(mr))
                data = MultiRegionObject(collect_data(mr.regions), devices(mr))
                bcs  = MultiRegionObject(collect_bcs(mr.regions),  devices(mr))
                return Field{loc[1], loc[2], loc[3]}(mrg, data, bcs, c.operand, c.status) 
            end
        end
    end
end

# function Statistics.mean(f::Function, c::MultiRegionField; kwargs...) 
#     mr = construct_regionally(Statistics.mean, f, c; kwargs...)
#     return Statistics.mean([r for r in mr.regions])
# end

Statistics.mean(c::MultiRegionField; kwargs...) = Statistics.mean(identity, c; kwargs...)

validate_reduction_location!(loc, p)            = nothing
validate_reduction_location!(loc, ::XPartition) = loc[1] == Nothing && error("Partial reductions across X with XPartition not supported yet")

collect_data(f::NTuple{N, <:Field}) where N = Tuple(f[i].data for i in 1:N)
collect_bcs(f::NTuple{N, <:Field})  where N = Tuple(f[i].boundary_conditions for i in 1:N)
collect_grid(f::NTuple{N, <:Field}) where N = Tuple(f[i].grid for i in 1:N)

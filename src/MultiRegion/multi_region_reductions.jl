using Statistics
import Statistics.mean
import Statistics.norm
import Statistics.dot

reductions = (:(Base.sum), :(Base.maximum), :(Base.minimum), :(Base.prod), :(Base.any), :(Base.all), :(Statistics.mean))

#####
##### Reductions are still veeeery slow on MultiRegionGrids. Avoid as much as possible!
#####

#####
##### Non allocating reductions are not implemented as of now
#####

# Allocating reductions
for reduction in reductions
    @eval begin
        function $(reduction)(f::Function, c::MultiRegionField; kwargs...)
            mr = construct_regionally($(reduction), f, c; kwargs...)
            if mr.regional_objects isa NTuple{<:Any, <:Number}
                return $(reduction)([r for r in mr.regional_objects]) 
            else
                FT   = eltype(first(mr.regional_objects))
                loc  = location(first(mr.regional_objects))
                validate_reduction_location!(loc, c.grid.partition)
                mrg  = MultiRegionGrid{FT, loc[1], loc[2], loc[3]}(architecture(c), c.grid.partition, MultiRegionObject(collect_grid(mr.regional_objects), devices(mr)), devices(mr))
                data = MultiRegionObject(collect_data(mr.regional_objects), devices(mr))
                bcs  = MultiRegionObject(collect_bcs(mr.regional_objects),  devices(mr))
                return Field{loc[1], loc[2], loc[3]}(mrg, data, bcs, c.operand, c.status) 
            end
        end
    end
end

Statistics.mean(c::MultiRegionField; kwargs...) = Statistics.mean(identity, c; kwargs...)

validate_reduction_location!(loc, p) = nothing
validate_reduction_location!(loc, ::XPartition) = loc[1] == Nothing && error("Partial reductions across X with XPartition are not supported yet")
validate_reduction_location!(loc, ::YPartition) = loc[2] == Nothing && error("Partial reductions across Y with YPartition are not supported yet")

collect_data(f::NTuple{N, <:Field}) where N = Tuple(f[i].data for i in 1:N)
collect_bcs(f::NTuple{N, <:Field})  where N = Tuple(f[i].boundary_conditions for i in 1:N)
collect_grid(f::NTuple{N, <:Field}) where N = Tuple(f[i].grid for i in 1:N)

const MRD = Union{MultiRegionField, MultiRegionObject}

# make it more efficient?
Statistics.dot(f::MRD,  g::MRD)  = sum([r for r in construct_regionally(dot, f, g).regional_objects])
Statistics.norm(f::MRD) = sqrt(dot(f, f))

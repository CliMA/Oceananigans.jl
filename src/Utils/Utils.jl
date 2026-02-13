module Utils

export configure_kernel, launch!, KernelParameters
export prettytime, pretty_filesize
export tupleit, parenttuple, datatuple, datatuples
export ordered_dict_show
export instantiate
export with_tracers
export versioninfo_with_gpu, oceananigans_versioninfo
export seconds_to_nanosecond, period_to_seconds, time_difference_seconds, add_time_interval
export TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes, AndSchedule, OrSchedule, ConsecutiveIterations
export apply_regionally!, construct_regionally, @apply_regionally, MultiRegionObject
export isregional, getregion, _getregion, regions, sync_device!
export newton_div, NormalDivision, ConvertingDivision, BackendOptimizedDivision
export TabulatedFunction

#####
##### Misc. small utils
#####

instantiate(T::Type) = T()
instantiate(t) = t

getnamewrapper(type) = typeof(type).name.wrapper

#####
##### Include utils
#####

include("prettysummary.jl")
include("kernel_launching.jl")
include("prettytime.jl")
include("pretty_filesize.jl")
include("tuple_utils.jl")
include("ordered_dict_show.jl")
include("with_tracers.jl")
include("versioninfo.jl")
include("times_and_datetimes.jl")
include("schedules.jl")
include("user_function_arguments.jl")
include("multi_region_transformation.jl")
include("sum_of_arrays.jl")
include("newton_div.jl")
include("tabulated_function.jl")

end # module

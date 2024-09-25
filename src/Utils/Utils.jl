module Utils

export configured_kernel, work_layout, launch!, KernelParameters
export prettytime, pretty_filesize
export tupleit, parenttuple, datatuple, datatuples
export validate_intervals, time_to_run
export ordered_dict_show
export instantiate
export with_tracers
export versioninfo_with_gpu, oceananigans_versioninfo
export TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes, AndSchedule, OrSchedule 
export apply_regionally!, construct_regionally, @apply_regionally, @regional, MultiRegionObject
export isregional, getregion, _getregion, getdevice, switch_device!, sync_device!, sync_all_devices!

import CUDA  # To avoid name conflicts

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
include("output_writer_diagnostic_utils.jl")
include("ordered_dict_show.jl")
include("with_tracers.jl")
include("versioninfo.jl")
include("schedules.jl")
include("user_function_arguments.jl")
include("multi_region_transformation.jl")
include("coordinate_transformations.jl")
include("sum_of_arrays.jl")

end # module

module Utils

export launch_config, work_layout, launch!
export cell_advection_timescale
export TimeStepWizard, update_Δt!
export prettytime, pretty_filesize
export tupleit, parenttuple, datatuple, datatuples
export validate_intervals, time_to_run
export ordered_dict_show
export with_tracers
export versioninfo_with_gpu, oceananigans_versioninfo
export instantiate
export TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes, AndSchedule, OrSchedule 
export apply_regionally!, construct_regionally, @apply_regionally, @regional
export isregional, getregion, getdevice, switch_device!, sync_device!

import CUDA  # To avoid name conflicts

#####
##### Misc. small utils
#####

instantiate(x) = x
instantiate(X::DataType) = X()

#####
##### Include utils
#####

include("prettysummary.jl")
include("kernel_launching.jl")
include("cell_advection_timescale.jl")
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

end

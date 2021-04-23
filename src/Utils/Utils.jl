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

import Oceananigans: short_show

#####
##### Misc. small utils
#####

instantiate(x) = x
instantiate(X::DataType) = X()

short_show(a) = string(a) # fallback
short_show(f::Function) = string(Symbol(f))

#####
##### Include utils
#####

include("kernel_launching.jl")
include("cell_advection_timescale.jl")
include("pretty_time.jl")
include("pretty_filesize.jl")
include("tuple_utils.jl")
include("output_writer_diagnostic_utils.jl")
include("ordered_dict_show.jl")
include("with_tracers.jl")
include("versioninfo.jl")
include("schedules.jl")
include("user_function_arguments.jl")

end

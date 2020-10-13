module Utils

export
    second, minute, hour, day, year, meter, kilometer,
    seconds, minutes, hours, days, years, meters, kilometers,
    KiB, MiB, GiB, TiB,
    launch_config, work_layout, launch!,
    cell_advection_timescale,
    TimeStepWizard, update_Δt!,
    prettytime, pretty_filesize,
    tupleit, parenttuple, datatuple, datatuples,
    validate_intervals, time_to_run,
    ordered_dict_show,
    with_tracers,
    versioninfo_with_gpu, oceananigans_versioninfo,
    instantiate

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

include("adapt_structure.jl")
include("units.jl")
include("automatic_halo_sizing.jl")
include("kernel_launching.jl")
include("cell_advection_timescale.jl")
include("pretty_time.jl")
include("pretty_filesize.jl")
include("tuple_utils.jl")
include("output_writer_diagnostic_utils.jl")
include("ordered_dict_show.jl")
include("with_tracers.jl")
include("versioninfo.jl")

end

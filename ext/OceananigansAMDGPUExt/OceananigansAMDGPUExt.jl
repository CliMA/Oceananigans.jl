module OceananigansAMDGPUExt

using AMDGPU

include("Architectures.jl")
include("Grids.jl")
include("Utils.jl")
include("BoundaryConditions.jl")
include("Fields.jl")
include("DistributedComputations.jl")
include("MultiRegion.jl")

using .Architectures
using .Grids
using .Utils
using .BoundaryConditions
using .Fields
using .DistributedComputations
using .MultiRegion

function __init__()
    if AMDGPU.has_rocm_gpu()
        @debug "ROCm-enabled GPU(s) detected:"
        for (id, agent) in enumerate(AMDGPU.devices())
            @debug "$id: $(agent.name)"
        end
    else
        @debug "No ROCm-enabled GPU was found"
    end
end

end # module
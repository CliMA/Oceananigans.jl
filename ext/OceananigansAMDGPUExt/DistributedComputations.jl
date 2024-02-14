module DistributedComputations

using ..Architectures: ROCArray, ROCmGPU

using Oceananigans.DistributedComputations: DistributedField

import Oceananigans.DistributedComputations: set!

function set!(u::DistributedField, v::ROCArray)
    gsize = global_size(architecture(u), size(u))

    if size(v) == size(u)
        f = arch_array(architecture(u), v)
        u .= f
        return u
    elseif size(v) == gsize
        f = partition_global_array(architecture(u), v, size(u))
        u .= f
        return u
    else
        throw(ArgumentError("ERROR: DimensionMismatch: array could not be set to match destination field"))
    end
end

end # module

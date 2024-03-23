module MultiRegion

using AMDGPU

using ..Architectures: ROCmGPU

import Oceananigans.MultiRegion: validate_devices

# multi_region_utils.jl

validate_devices(p, ::ROCmGPU, ::Nothing) = 1

function validate_devices(partition, ::ROCmGPU, devices)
    @assert length(unique(devices)) ≤ length(AMDGPU.devices())
    @assert maximum(devices) ≤ length(AMDGPU.devices())
    @assert length(devices) ≤ length(partition)
    return devices
end

function validate_devices(partition, ::ROCmGPU, devices::Number)
    @assert devices ≤ length(AMDGPU.devices())
    @assert devices ≤ length(partition)
    return devices
end

end # module

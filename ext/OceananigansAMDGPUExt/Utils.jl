module Utils

using ..Architectures: ROCArray, ROCmGPU 

using AMDGPU

import Oceananigans.Utils: getdevice, switch_device!, sync_device!

# multi_region_transformations.jl

const ROCmGPUVar = Union{ROCArray, HIPContext, Ptr}

@inline getdevice(roc::ROCmGPUVar, i) = AMDGPU.device(roc)
@inline getdevice(roc::ROCmGPUVar) = AMDGPU.device(roc)

@inline switch_device!(dev::HIPDevice) = AMDGPU.default_device!(dev)

@inline sync_device!(::ROCmGPU)  = AMDGPU.synchronize()
@inline sync_device!(::HIPDevice) = AMDGPU.synchronize()

end # module

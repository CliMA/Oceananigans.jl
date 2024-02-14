module Architectures

using AMDGPU
using Oceananigans

export ROCmGPU

import Oceananigans.Architectures: device, architecture, array_type, arch_array,
                                   device_copy_to!, unsafe_free!

const ROCmGPU = GPU{<:AMDGPU.ROCBackend}
ROCmGPU() = GPU(AMDGPU.ROCBackend())

device(::ROCmGPU) = AMDGPU.ROCBackend()
architecture(::ROCArray) = GPU(AMDGPU.ROCBackend())

array_type(::ROCmGPU) = ROCArray

arch_array(::CPU, a::ROCArray) = Array(a)

arch_array(::ROCmGPU, a::Array) = ROCArray(a)
arch_array(::ROCmGPU, a::ROCArray) = a
arch_array(::ROCmGPU, a::BitArray) = ROCArray(a)
arch_array(::ROCmGPU, a::SubArray{<:Any, <:Any, <:Array}) = ROCArray(a)

@inline function device_copy_to!(dst::ROCArray, src::ROCArray; async::Bool = false) 
    AMDGPU.mem.transfer!(dst.buf, src.buf, sizeof(src))
    return dst
end

@inline unsafe_free!(a::ROCArray) = AMDGPU.unsafe_free!(a)

end # module

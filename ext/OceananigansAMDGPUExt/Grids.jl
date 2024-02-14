module Grids

using AMDGPU
using ..Architectures: ROCmGPU

import Base: zeros

zeros(FT, ::ROCmGPU, N...) = AMDGPU.zeros(FT, N...)

end # module

using CUDA: CuArray
using Oceananigans.Architectures: AbstractCPUArchitecture, AbstractGPUArchitecture

import Base: zeros

zeros(FT, ::AbstractCPUArchitecture, N...) = zeros(FT, N...)

function zeros(FT, ::AbstractGPUArchitecture, N...)
    a = CuArray{FT}(undef, N...)
    a .= 0
    return a
end

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)

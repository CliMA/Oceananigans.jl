using CUDA: CuArray
using Oceananigans.Architectures: CPU, GPU

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)

function zeros(FT, ::GPU, N...)
    a = CuArray{FT}(undef, N...)
    a .= 0
    return a
end

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)

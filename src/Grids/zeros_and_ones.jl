using CUDA
using AMDGPU
using Oceananigans.Architectures: CPU, GPU, AbstractArchitecture

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, ::CUDAGPU, N...) = CUDA.zeros(FT, N...)
zeros(FT, ::ROCmGPU, N...) = AMDGPU.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid) = one(eltype(grid))

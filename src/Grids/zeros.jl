using CUDA
using AMDGPU
using Oceananigans.Architectures: CPU, CUDAGPU, AMDGPU, AbstractMultiArchitecture

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, ::CUDAGPU, N...) = CUDA.zeros(FT, N...)
zeros(FT, ::ROCMGPU, N...) = AMDGPU.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

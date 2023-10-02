using CUDA
using Metal
using Oceananigans.Architectures: CPU, GPU, MetalBackend, AbstractArchitecture

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, ::GPU{<:CUDABackend}, N...) = CUDA.zeros(FT, N...)
zeros(FT, ::GPU{<:MetalBackend}, N...) = Metal.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid) = one(eltype(grid))

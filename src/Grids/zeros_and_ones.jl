using CUDA
using Metal

using Oceananigans.Architectures: device, AbstractArchitecture, AbstractSerialArchitecture

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, ::GPU{<:CUDA.CUDABackend}, N...)   = CUDA.zeros(FT, N...)
zeros(FT, ::GPU{<:Metal.MetalBackend}, N...) = Metal.zeros(FT, N...)
zeros(::Float64, ::GPU{<:Metal.MetalBackend}, N...) = error("Metal does not support Float64 arrays")

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid) = one(eltype(grid))

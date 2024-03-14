using CUDA
using Oceananigans.Architectures: CPU, GPU, AbstractArchitecture, device
using KernelAbstractions

import Base: zeros

zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, arch::AbstractArchitecture, N...) = KernelAbstractions.zeros(device(arch), FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid) = one(eltype(grid))

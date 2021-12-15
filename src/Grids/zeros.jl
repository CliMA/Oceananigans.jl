using CUDA
using Oceananigans.Architectures: CPU, GPU, AbstractMultiArchitecture

import Base: zeros



zeros(FT, ::CPU, N...) = zeros(FT, N...)
zeros(FT, ::GPU, N...) = CUDA.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)

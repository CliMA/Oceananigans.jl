using Oceananigans.Architectures: device, AbstractArchitecture

import KernelAbstractions: zeros

zeros(FT, arch::AbstractArchitecture, N...) = zeros(device(arch), FT, N...)

zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid)  = one(eltype(grid))

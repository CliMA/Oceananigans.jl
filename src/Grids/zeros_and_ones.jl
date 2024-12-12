import KernelAbstractions: zeros
using Oceananigans.Architectures: device, AbstractArchitecture, AbstractSerialArchitecture

zeros(FT, arch::AbstractSerialArchitecture, N...) = zeros(device(arch), FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
zeros(grid::AbstractGrid, N...) = zeros(eltype(grid), architecture(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid) = one(eltype(grid))

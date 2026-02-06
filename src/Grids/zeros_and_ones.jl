using Oceananigans.Architectures: CPU, AbstractArchitecture
using Oceananigans.Architectures: device, AbstractArchitecture

import KernelAbstractions

unwrapped_eltype(::Type{T}) where {T} = T

Base.zeros(arch::AbstractArchitecture, FT, N...) = KernelAbstractions.zeros(device(arch), unwrapped_eltype(FT), N...)
Base.zeros(grid::AbstractGrid, N...) = zeros(architecture(grid), eltype(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid)  = one(eltype(grid))

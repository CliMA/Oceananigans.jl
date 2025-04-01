using Oceananigans.Architectures: CPU, GPU, AbstractArchitecture
using Oceananigans.Architectures: device, AbstractArchitecture

import KernelAbstractions
import Base: zeros

unwrapped_eltype(::Type{T}) where {T} = T

zeros(arch::AbstractArchitecture, FT, N...) = KernelAbstractions.zeros(device(arch), unwrapped_eltype(FT), N...)
zeros(grid::AbstractGrid, N...) = zeros(architecture(grid), eltype(grid), N...)

@inline Base.zero(grid::AbstractGrid) = zero(eltype(grid))
@inline Base.one(grid::AbstractGrid)  = one(eltype(grid))


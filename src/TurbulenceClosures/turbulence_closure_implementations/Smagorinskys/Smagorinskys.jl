module Smagorinskys

using Oceananigans.Operators
using Oceananigans.Fields

using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

export Smagorinsky
export DirectionallyAveragedCoefficient
export LillyCoefficient

include("smagorinsky.jl")
include("scale_invariant_operators.jl")

end

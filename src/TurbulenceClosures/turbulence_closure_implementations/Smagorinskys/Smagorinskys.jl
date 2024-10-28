module Smagorinskys

using Oceananigans.Operators
using Oceananigans.Fields

using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

export Smagorinsky
export SmagorinskyLilly
export DynamicCoefficient
export LagrangianAveraging
export LillyCoefficient

include("smagorinsky.jl")
include("dynamic_coefficient.jl")
include("lilly_coefficient.jl")
include("scale_invariant_operators.jl")

end

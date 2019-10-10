module AbstractOperations

export ∂x, ∂y, ∂z

using Base: @propagate_inbounds

using Oceananigans
using Oceananigans: Face, Cell, AbstractLocatedField, device, launch_config, architecture
import Oceananigans: totaldata

import Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                        ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf

using GPUifyLoops: @launch, @loop

import Base: *, -, +, /, getindex

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

totaldata(op::AbstractOperation) = op

include("binary_operations.jl")
include("derivatives.jl")
include("computations.jl")

end # module

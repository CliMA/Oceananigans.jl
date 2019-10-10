module AbstractOperations

export ∂x, ∂y, ∂z

using Base: @propagate_inbounds

using Oceananigans

using Oceananigans: AbstractModel, AbstractLocatedField, Face, Cell, 
                    device, launch_config, architecture,
                    HorizontalAverage, zero_halo_regions!, normalize_horizontal_sum!

import Oceananigans: data, architecture

import Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                        ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf

using GPUifyLoops: @launch, @loop

import Base: *, -, +, /, getindex

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

data(op::AbstractOperation) = op
Base.parent(op::AbstractOperation) = op

include("binary_operations.jl")
include("derivatives.jl")
include("computations.jl")

end # module

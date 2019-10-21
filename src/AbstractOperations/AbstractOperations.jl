module AbstractOperations

export ∂x, ∂y, ∂z, @at, Computation, compute!, @unary

using Base: @propagate_inbounds

using Oceananigans: @hascuda

@hascuda using CUDAnative, CUDAdrv, CuArrays

using Oceananigans, Adapt

using Oceananigans: AbstractModel, AbstractGrid, AbstractField, AbstractLocatedField, Face, Cell, 
                    xnode, ynode, znode, location, show_location, show_domain, short_show,
                    device, launch_config, architecture, 
                    HorizontalAverage, zero_halo_regions!, normalize_horizontal_sum!

import Oceananigans: run_diagnostic, data, architecture

using Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                       ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf,
                                       ▶xy_cca, ▶xy_ffa, ▶xy_cfa, ▶xy_fca, 
                                       ▶xz_cac, ▶xz_faf, ▶xz_caf, ▶xz_fac, 
                                       ▶yz_acc, ▶yz_aff, ▶yz_acf, ▶yz_afc,
                                       ▶xyz_ccc, ▶xyz_fcc, ▶xyz_cfc, ▶xyz_ccf,
                                       ▶xyz_fff, ▶xyz_ffc, ▶xyz_fcf, ▶xyz_cff

using GPUifyLoops: @launch, @loop

#####
##### Basic functionality
#####

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

const ALF = AbstractLocatedField

data(op::AbstractOperation) = op
Base.parent(op::AbstractOperation) = op

# AbstractOperation macros add their associated functions to this list
const operators = []

include("function_fields.jl")
include("interpolation_utils.jl")
include("grid_validation.jl")

include("unary_operations.jl")
include("binary_operations.jl")
#include("polynary_operations.jl")
include("derivatives.jl")

include("computations.jl")
include("show_abstract_operations.jl")

# Make some operators!
import Base: sqrt, sin, cos, exp, tanh

@unary sqrt sin cos exp tanh
@unary -

#=
@binary +
@binary -
@binary /
@binary *
@binary ^
=#

end # module

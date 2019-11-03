module Operators

using Oceananigans

export
    FU, FV, FW, FC,
    RU, RV, RW, Rρ, RC

include("areas_and_volumes.jl")
include("difference_operators.jl")
include("derivative_operators.jl")
include("interpolation_operators.jl")
include("divergence_operators.jl")
include("laplacian_operators.jl")

include("compressible_operators.jl")

end

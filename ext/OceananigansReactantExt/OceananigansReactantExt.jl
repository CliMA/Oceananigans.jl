module OceananigansReactantExt

using Reactant
using Oceananigans
using OffsetArrays

deconcretize(obj) = obj # fallback
deconcretize(a::OffsetArray) = OffsetArray(Array(a.parent), a.offsets...)

include("Utils.jl")
using .Utils

include("Architectures.jl")
using .Architectures

include("Grids.jl")
using .Grids

include("Fields.jl")
using .Fields

include("TimeSteppers.jl")
using .TimeSteppers

include("Models.jl")
using .Models

include("Simulations/Simulations.jl")
using .Simulations

#####
##### Telling Reactant how to construct types
#####

import ConstructionBase: constructorof 

constructorof(::Type{<:RectilinearGrid{FT, TX, TY, TZ}}) where {FT, TX, TY, TZ} = RectilinearGrid{TX, TY, TZ}
constructorof(::Type{<:VectorInvariant{N, FT, M}}) where {N, FT, M} = VectorInvariant{N, FT, M}

# These are additional modules that may need to be Reactantified in the future:
#
# include("Utils.jl")
# include("BoundaryConditions.jl")
# include("Fields.jl")
# include("MultiRegion.jl")
# include("Solvers.jl")
#
# using .Utils
# using .BoundaryConditions
# using .Fields
# using .MultiRegion
# using .Solvers

end # module


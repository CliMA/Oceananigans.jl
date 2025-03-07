module OceananigansReactantExt

using Reactant
using OffsetArrays

deconcretize(obj) = obj # fallback
deconcretize(a::OffsetArray) = OffsetArray(Array(a.parent), a.offsets...)

include("Architectures.jl")
using .Architectures

include("Grids.jl")
using .Grids

include("Fields.jl")
using .Fields

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

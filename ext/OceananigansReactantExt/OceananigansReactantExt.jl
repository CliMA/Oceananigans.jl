module OceananigansReactantExt

using Reactant

include("Architectures.jl")
using .Architectures


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

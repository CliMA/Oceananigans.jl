import MPI

using Oceananigans

struct DistributedModel{A, R, G, C}
                 ranks :: R
                models :: A
    connectivity_graph :: G
              MPI_Comm :: C
end

const FieldBoundaryConditions = NamedTuple{(:east, :west, :north, :south, :top, :bottom)}

function validate_tupled_argument(arg, argtype, argname)
    length(arg) == 3        || throw(ArgumentError("length($argname) must be 3."))
    all(isa.(arg, argtype)) || throw(ArgumentError("$argname=$arg must contain $argtype s."))
    all(arg .> 0)           || throw(ArgumentError("Elements of $argname=$arg must be > 0!"))
    return nothing
end

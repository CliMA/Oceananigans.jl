module Operators

export
    δx!,
    δy!,
    δz!,
    avgx!,
    avgy!,
    avgz!,
    div!,
    div_flux!,
    u∇u!,
    u∇v!,
    u∇w!,
    κ∇²!

include("ops_regular_cartesian_grid.jl")

end

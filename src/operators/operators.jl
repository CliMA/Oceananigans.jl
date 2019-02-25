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
	div_flux_4!,
    u∇u!,
    u∇v!,
    u∇w!,
    κ∇²!,
    𝜈∇²u!,
    𝜈∇²v!,
    𝜈∇²w!,
    ∇²_ppn!

include("ops_regular_cartesian_grid.jl")

end

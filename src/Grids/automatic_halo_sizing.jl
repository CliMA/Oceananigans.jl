"""
    required_halo_size(tendency_term)

Returns the required size of halos for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Grids: required_halo_size

required_halo_size(CenteredFourthOrder())

# output
2
"""

function required_halo_size end

required_halo_size(nothing) = 1

inflat_halo_size_one_dimension(H, Hx, TX          ) = max(H, Hx)
inflat_halo_size_one_dimension(H, Hx, ::Type{Flat}) = 0

function inflate_halo_size(Hx, Hy, Hz, topology, tendency_terms...)

    oldHx = Hx
    oldHy = Hy
    oldHz = Hz

    for term in tendency_terms
         H = required_halo_size(term)
        Hx = inflat_halo_size_one_dimension(H, Hx, topology[1])
        Hy = inflat_halo_size_one_dimension(H, Hy, topology[2])
        Hz = inflat_halo_size_one_dimension(H, Hz, topology[3])
    end

    if Hx != oldHx || Hy != oldHy || Hz != oldHz
        @warn "Inflating model grid halo size to ($Hx, $Hy, $Hz) and recreating grid. " *
        "The model grid will be different from the input grid. To avoid this warning, " *
        "pass halo=($Hx, $Hy, $Hz) when constructing the grid."
    end

    return Hx, Hy, Hz
end

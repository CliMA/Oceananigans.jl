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

required_halo_size(tendency_term) = 1

inflate_halo_size_one_dimension(req_H, old_H, _, grid)            = max(req_H, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}, grid) = 0

function inflate_halo_size(Hx, Hy, Hz, grid, tendency_terms...)
    topo = topology(grid)
    for term in tendency_terms
         H = required_halo_size(term)
        Hx = inflate_halo_size_one_dimension(H, Hx, topo[1], grid)
        Hy = inflate_halo_size_one_dimension(H, Hy, topo[2], grid)
        Hz = inflate_halo_size_one_dimension(H, Hz, topo[3], grid)
    end

    return Hx, Hy, Hz
end

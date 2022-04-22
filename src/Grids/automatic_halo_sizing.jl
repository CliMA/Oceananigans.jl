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

inflate_halo_size_one_dimension(req_H, old_H, _)            = max(req_H, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}) = 0

function inflate_halo_size(Hx, Hy, Hz, topology, tendency_terms...)
    for term in tendency_terms
         H = required_halo_size(term)
        Hx = inflate_halo_size_one_dimension(H, Hx, topology[1])
        Hy = inflate_halo_size_one_dimension(H, Hy, topology[2])
        Hz = inflate_halo_size_one_dimension(H, Hz, topology[3])
    end

    return Hx, Hy, Hz
end

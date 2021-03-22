"""
    required_halo_size(tendency_term)

Returns the required size of halos for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Utils: required_halo_size

required_halo_size(CenteredFourthOrder())

# output
2
"""
required_halo_size(anything) = 1

inflat_halo_size_one_dimension(H, Hx, TX          ) = max(H, Hx)
inflat_halo_size_one_dimension(H, Hx, ::Type{Flat}) = 0

function inflate_halo_size(Hx, Hy, Hz, topology, tendency_terms...)
    for term in tendency_terms
        H = required_halo_size(term)

        Hx = inflat_halo_size_one_dimension(H, Hx, topology[1])
        Hy = inflat_halo_size_one_dimension(H, Hy, topology[2])
        Hz = inflat_halo_size_one_dimension(H, Hz, topology[3])
    end
    return Hx, Hy, Hz
end

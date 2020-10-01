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

function inflate_halo_size(Hx, Hy, Hz, tendency_terms...)
    for term in tendency_terms
        H = required_halo_size(term)

        Hx = max(Hx, H)
        Hy = max(Hy, H)
        Hz = max(Hz, H)
    end

    return Hx, Hy, Hz
end

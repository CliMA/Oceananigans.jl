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

#FJP: this should depend on the topology!!!!
#using Oceananigans.Grids: topology    (maybe in shallow_water_model.jl?)
#in max calculations, ignore if Flat

function inflate_halo_size(Hx, Hy, Hz, topology, tendency_terms...)
    for term in tendency_terms
        H = required_halo_size(term)
        if topology[1] == Flat
            Hx = 0
        else
            Hx = max(Hx,H)
        end
        if topology[2] == Flat
            Hy = 0
        else
            Hy = max(Hy,H)
        end
        if topology[3] == Flat
            Hz = 0
        else
            Hz = max(Hz,H)
        end
    end

    return Hx, Hy, Hz
end



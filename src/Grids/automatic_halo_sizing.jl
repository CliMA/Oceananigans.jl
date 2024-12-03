"""
    required_halo_size_x(tendency_term)

Return the required size of halos in the x direction for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Grids: required_halo_size_x

required_halo_size_x(CenteredFourthOrder())

# output
2
"""
function required_halo_size_x end

"""
    required_halo_size_y(tendency_term)

Return the required size of halos in the y direction for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Grids: required_halo_size_y

required_halo_size_y(CenteredFourthOrder())

# output
2
"""
function required_halo_size_y end

"""
    required_halo_size_z(tendency_term)

Return the required size of halos in the y direction for a term appearing
in the tendency for a velocity field or tracer field.

Example
=======

```jldoctest
using Oceananigans.Advection: CenteredFourthOrder
using Oceananigans.Grids: required_halo_size_z

required_halo_size_z(CenteredFourthOrder())

# output
2
"""
function required_halo_size_z end

required_halo_size_x(tendency_term) = 1
required_halo_size_x(::Nothing) = 0
required_halo_size_y(tendency_term) = 1
required_halo_size_y(::Nothing) = 0
required_halo_size_z(tendency_term) = 1
required_halo_size_z(::Nothing) = 0

inflate_halo_size_one_dimension(req_H, old_H, _, grid)            = max(req_H, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}, grid) = 0

function inflate_halo_size(Hx, Hy, Hz, grid, tendency_terms...)
    topo = topology(grid)
    for term in tendency_terms
        Hx_required = required_halo_size_x(term)
        Hy_required = required_halo_size_y(term)
        Hz_required = required_halo_size_z(term)
        Hx = inflate_halo_size_one_dimension(Hx_required, Hx, topo[1], grid)
        Hy = inflate_halo_size_one_dimension(Hy_required, Hy, topo[2], grid)
        Hz = inflate_halo_size_one_dimension(Hz_required, Hz, topo[3], grid)
    end

    return Hx, Hy, Hz
end

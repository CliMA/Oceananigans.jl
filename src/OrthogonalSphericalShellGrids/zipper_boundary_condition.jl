using Oceananigans.Grids: Center, Face
using Oceananigans.Utils: KernelParameters, launch!

import Oceananigans.BoundaryConditions: bc_str, _fill_north_halo!, apply_y_north_bc!
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification, BoundaryCondition
import Oceananigans.Fields: validate_boundary_condition_location

struct Zipper <: AbstractBoundaryConditionClassification end

"""
    ZipperBoundaryCondition(sign = 1)

Create a zipper boundary condition specific to the `TripolarGrid`.
A Zipper boundary condition is similar to a periodic boundary condition, but, 
instead of retrieving the value from the opposite boundary, it splits the boundary 
in two and retrieves the value from the opposite side of the boundary. 
It is possible to think of it as a periodic boundary over a folded domain.

When copying in halos, folded velocities need to switch sign, while tracers or similar fields do not.

Note: the Tripolar boundary condition is particular because the last grid point 
at the north edge is repeated for tracers. For this reason we do not start copying the halo 
from the last grid point but from the second to last grid point.

Example
=======

Consider the northern edge of a tripolar grid where P indicates the j - location of the poles

```
                  P                         P
                  |            |            |            |            
 Ny (center) ->   u₁    c₁     u₂    c₂     u₃    c₃     u₄    c₄     
 Ny (face)   ->   |---- v₁ ----|---- v₂ ----|---- v₃ ----|---- v₄ ----
                                                      Nx    Nx
                                                    (face) (center)
```
The grid ends at `Ny` because it is periodic in nature, but, given the fold, 
```
c₁ == c₄
```

and
```
c₂ == c₃
```

This is not the case for the ``v``-velocity (or any field on the `j`-faces), where
the last grid point is not repeated. Because of this redundancy, we ensure consistency
by substituting the redundant part of fields Centered in ``x``, in the last row.
"""
ZipperBoundaryCondition(sign = 1) = BoundaryCondition(Zipper(), sign)

const ZBC = BoundaryCondition{<:Zipper}

bc_str(zip::ZBC) = "Zipper"

validate_boundary_condition_location(bc::Zipper, loc::Center, side) = 
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))

validate_boundary_condition_location(bc::Zipper, loc::Face, side) = 
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))

@inline apply_y_north_bc!(Gc, loc, ::ZBC, args...) = nothing
    
#####
##### Outer functions for filling halo regions for Zipper boundary conditions.
#####

@inline function fold_north_face_face!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    sign = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@inline function fold_north_face_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    sign  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    # We substitute the redundant part of the last row to ensure consistency
    @inbounds c[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * c[i′, Ny, k], c[i, Ny, k])

    return nothing
end

@inline function fold_north_center_face!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@inline function fold_north_center_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    # We substitute the redundant part of the last row to ensure consistency
    @inbounds c[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * c[i′, Ny, k], c[i, Ny, k])

    return nothing
end

const CCLocation = Tuple{<:Center, <:Center, <:Any} 
const FCLocation = Tuple{<:Face,   <:Center, <:Any} 
const CFLocation = Tuple{<:Center, <:Face,   <:Any} 
const FFLocation = Tuple{<:Face,   <:Face,   <:Any} 

# tracers or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::ZBC, ::CCLocation, args...) = fold_north_center_center!(i, k, grid, bc.condition, c)

# u-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, u, bc::ZBC, ::FCLocation, args...) = fold_north_face_center!(i, k, grid, bc.condition, u)

# v-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, v, bc::ZBC, ::CFLocation, args...) = fold_north_center_face!(i, k, grid, bc.condition, v)

# vorticity or similar fields
@inline _fill_north_halo!(i, k, grid, ζ, bc::ZBC, ::FFLocation, args...) = fold_north_face_face!(i, k, grid, bc.condition, ζ)

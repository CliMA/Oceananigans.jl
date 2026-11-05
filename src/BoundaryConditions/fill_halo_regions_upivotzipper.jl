using Oceananigans.Grids: Center, Face
using Oceananigans.Utils: KernelParameters, launch!

"""
    UPivotZipperBoundaryCondition(sign = 1)

Create a U-point pivot zipper boundary condition specific to a `TripolarGrid`.
This zipper BC folds the northern boundary on itself along the `XFace` line,
with a pivot point located at a `(Face, Center)` location (a U-point pivot).

When copying in halos, folded velocities need to switch sign, while tracers or similar fields do not.

Note: Two types of zipper boundary conditions are currently implemented:
- F-point pivot
- U-point pivot (this one)

Note: this Tripolar boundary condition is particular because the last grid point
at the north edge is repeated for tracers. For this reason we do not start copying the halo
from the last grid point but from the second to last grid point.

Example
=======

Consider the northern edge of a tripolar grid where P indicates the pivot point,
then there must be a 180° rotation symmetry around the pivot point:
```
                    │            │            │            │            │
Ny + 1 (center) ─▶ -u₂    c₆    -u₅    c₅    -u₄    c₄    -u₃    c₃    -u₂
                    │            │            │            │            │
Ny + 1 (face)   ─▶  ├─── -v₄ ────┼─── -v₃ ────┼─── -v₂ ────┼─── -v₁ ────┤
                    │            │            │            │            │
Ny     (center) ─▶  0     c₁     u₁    c₂     P     c₂    -u₁    c₁     0 ◀─ Fold
                    │            │            │            │            │
Ny     (face)   ─▶  ├──── v₁ ────┼──── v₂ ────┼──── v₃ ────┼──── v₄ ────┤
│                   │            │            │            │            │
Ny - 1 (center) ─▶  u₂    c₃     u₃    c₄     u₄    c₅     u₅    c₆     u₂
                    │            │            │            │            │
                                                           ▲     ▲
                                                           Nx    Nx
                                                         (face) (center)
```
"""

#####
##### Outer functions for filling halo regions for Zipper{UPivot} and Zipper{TPivot} boundary conditions,
##### shifted by `s` columns with respect to the `UPivot` mirror.
#####

@inline function fold_north_face_face_crow_pivot!(i, k, grid, sign, ζ, s)
    Nx, Ny, _ = size(grid)
    i′ = mod1(Nx - i + 2 + s, Nx) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            ζ[i, Ny + j, k] = sign * ζ[i′, Ny - j + 1, k]
        end
    end

    return nothing
end

@inline function fold_north_face_center_crow_pivot!(i, k, grid, sign, u, s)
    Nx, Ny, _ = size(grid)
    i′ = mod1(Nx - i + 2 + s, Nx) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            u[i, Ny + j, k] = sign * u[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    # We substitute the redundant part of the last row to ensure consistency
    @inbounds u[i, Ny, k] = ifelse(i > i′, sign * u[i′, Ny, k], u[i, Ny, k])

    return nothing
end

@inline function fold_north_center_face_crow_pivot!(i, k, grid, sign, v, s)
    Nx, Ny, _ = size(grid)

    i′ = mod1(Nx - i + 1 + s, Nx)
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            v[i, Ny + j, k] = sign * v[i′, Ny - j + 1, k]
        end
    end

    return nothing
end

@inline function fold_north_center_center_crow_pivot!(i, k, grid, sign, c, s)
    Nx, Ny, _ = size(grid)

    i′ = mod1(Nx - i + 1 + s, Nx)
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    # We substitute the redundant part of the last row to ensure consistency
    @inbounds c[i, Ny, k] = ifelse(i > i′, sign * c[i′, Ny, k], c[i, Ny, k])

    return nothing
end

const CCLocation = Tuple{<:Center, <:Center, <:Any}
const FCLocation = Tuple{<:Face,   <:Center, <:Any}
const CFLocation = Tuple{<:Center, <:Face,   <:Any}
const FFLocation = Tuple{<:Face,   <:Face,   <:Any}

const CenterRowZipper{P} = BoundaryCondition{<:Zipper{P}} where P <: Union{UPivot, TPivot}

# tracers or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::CenterRowZipper{P}, ::CCLocation, args...) where P = fold_north_center_center_crow_pivot!(i, k, grid, bc.condition, c, pivot_shift(P))

# u-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, u, bc::CenterRowZipper{P}, ::FCLocation, args...) where P = fold_north_face_center_crow_pivot!(i, k, grid, bc.condition, u, pivot_shift(P))

# v-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, v, bc::CenterRowZipper{P}, ::CFLocation, args...) where P = fold_north_center_face_crow_pivot!(i, k, grid, bc.condition, v, pivot_shift(P))

# vorticity or similar fields
@inline _fill_north_halo!(i, k, grid, ζ, bc::CenterRowZipper{P}, ::FFLocation, args...) where P = fold_north_face_face_crow_pivot!(i, k, grid, bc.condition, ζ, pivot_shift(P))

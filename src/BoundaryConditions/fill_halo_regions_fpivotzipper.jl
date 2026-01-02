using Oceananigans.Grids: Center, Face
using Oceananigans.Utils: KernelParameters, launch!

"""
    FPivotZipperBoundaryCondition(sign = 1)

Create a zipper boundary condition specific to the `TripolarGrid`.
A Zipper boundary condition is similar to a periodic boundary condition, but,
instead of retrieving the value from the opposite boundary, it splits the boundary
in two and retrieves the value from the opposite side of the boundary.
It is possible to think of it as a periodic boundary over a folded domain.

When copying in halos, folded velocities need to switch sign, while tracers or similar fields do not.

Note: There are two types of zipper boundary conditions:
F-point zipper (this one) and T-point zipper (original/default).
The F-point zipper folds on the cells' YFace, while the T-point zipper folds on the cells Center and XFace.

Example
=======

Consider the northern edge of a tripolar grid where P indicates the i - location of the poles

```
                    P                         P
                    |            |            |            |
 Ny (center)     -> u₁    c₁     u₂    c₂     u₃    c₃     u₄    c₄
 Ny (face)       -> |---- v₁ ----|---- v₂ ----|---- v₃ ----|---- v₄ ---- <- Fold
 Ny - 1 (center) -> u₁    c₁     u₂    c₂     u₃    c₃     u₄    c₄
 Ny - 1 (face)   -> |---- v₁ ----|---- v₂ ----|---- v₃ ----|---- v₄ ---- <- Fold
                                                           Nx    Nx
                                                         (face) (center)
```
The grid ends at `Ny` because it is periodic in nature, but, given the fold,
```
v₁ == -v₄
```

and
```
v₂ == -v₃
```

And the last row
"""

#####
##### Outer functions for filling halo regions for FPivotZipper boundary conditions.
#####

@inline function fold_north_face_face_fpivot!(i, k, grid, sign, ζ)
    Nx, Ny, _ = size(grid)
    i′ = Nx - i + 2 # Element Nx + 1 does not exist?
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            ζ[i, Ny + j, k] = sign * ζ[i′, Ny - j, k]
        end
    end

    # We substitute the redundant part of the last row of v to ensure consistency
    @inbounds ζ[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * ζ[i′, Ny, k], ζ[i, Ny, k])

    return nothing
end

@inline function fold_north_face_center_fpivot!(i, k, grid, sign, u)
    Nx, Ny, _ = size(grid)
    i′ = Nx - i + 2 # Element Nx + 1 does not exist?
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j = 0 : Hy
        @inbounds begin
            u[i, Ny + j, k] = sign * u[i′, Ny - j - 1, k]
        end
    end

    return nothing
end

@inline function fold_north_center_face_fpivot!(i, k, grid, sign, v)
    Nx, Ny, _ = size(grid)

    i′ = Nx - i + 1
    Hy = grid.Hy

    for j = 1 : Hy
        @inbounds begin
            v[i, Ny + j, k] = sign * v[i′, Ny - j, k]
        end
    end

    # We substitute the redundant part of the last row of v to ensure consistency
    @inbounds v[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * v[i′, Ny, k], v[i, Ny, k])

    return nothing
end

@inline function fold_north_center_center_fpivot!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)

    i′ = Nx - i + 1
    Hy = grid.Hy

    for j = 0 : Hy # The Ny line is duplicated so we substitute starting Ny-1
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j - 1, k]
        end
    end

    return nothing
end

const CCLocation = Tuple{<:Center, <:Center, <:Any}
const FCLocation = Tuple{<:Face,   <:Center, <:Any}
const CFLocation = Tuple{<:Center, <:Face,   <:Any}
const FFLocation = Tuple{<:Face,   <:Face,   <:Any}

# tracers or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::FZBC, ::CCLocation, args...) = fold_north_center_center_fpivot!(i, k, grid, bc.condition, c)

# u-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, u, bc::FZBC, ::FCLocation, args...) = fold_north_face_center_fpivot!(i, k, grid, bc.condition, u)

# v-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, v, bc::FZBC, ::CFLocation, args...) = fold_north_center_face_fpivot!(i, k, grid, bc.condition, v)

# vorticity or similar fields
@inline _fill_north_halo!(i, k, grid, ζ, bc::FZBC, ::FFLocation, args...) = fold_north_face_face_fpivot!(i, k, grid, bc.condition, ζ)


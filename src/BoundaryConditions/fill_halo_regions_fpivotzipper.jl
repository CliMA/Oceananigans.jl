using Oceananigans.Grids: Center, Face
using Oceananigans.Utils: KernelParameters, launch!

"""
    FPivotZipperBoundaryCondition(sign = 1)

Create a F-point pivot zipper boundary condition specific to a `TripolarGrid`.
This zipper BC folds the northern boundary on itself along the `YFace` line,
with a pivot point located at a `(Face, Face)` location (a F-point pivot).

When copying in halos, folded velocities need to switch sign, while tracers or similar fields do not.

Note: Two types of zipper boundary conditions are currently implented:
- F-point pivot (this one)
- U-point pivot

Example
=======

Consider the northern edge of a tripolar grid where P indicates the pivot point,
then there must be a 180° rotation symmetry around the pivot point:
```
                    │            │            │            │            │
Ny + 1 (face)   ─▶  ├─── -v₆ ────┼─── -v₅ ────┼─── -v₄ ────┼─── -v₃ ────┤
                    │            │            │            │            │
Ny     (center) ─▶ -u₁    c₄    -u₄    c₃    -u₃    c₂    -u₂    c₁    -u₁
                    │            │            │            │            │
Ny     (face)   ─▶  ├──── v₁ ────┼──── v₂ ─── P ── -v₂ ────┼─── -v₁ ────┤ ◀─ Fold
                    │            │            │            │            │
Ny - 1 (center) ─▶  u₁    c₁     u₂    c₂     u₃    c₃     u₄    c₄     u₁
                    │            │            │            │            │
Ny - 1 (face)   ─▶  ├──── v₃ ────┼──── v₄ ────┼──── v₅ ────┼──── v₆ ────┤
                    │            │            │            │            │
                                                           ▲     ▲
                                                           Nx    Nx
                                                         (face) (center)
```

Note that for the `RightFaceFolded` topology used here,
`YFaceField`s have an extra row (size `Ny` in the y-direction)
because the `v` velocities along the fold must be defined.

There is also an extra row for tracers and u-velocities which will be computed
dynamically but is redundant and will be eventually subsitituted by the boundary condition.
"""

#####
##### Outer functions for filling halo regions for Zipper{FPivot} boundary conditions.
#####

@inline function fold_north_face_face_fpivot!(i, k, grid, sign, ζ)
    Nx, Ny, _ = size(grid)
    i′ = Nx - i + 2 # Element Nx + 1 does not exist?
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j in 1:Hy
        @inbounds begin
            ζ[i, Ny + j, k] = sign * ζ[i′, Ny - j, k]
        end
    end

    # We substitute the redundant part of the last row of ζ to ensure consistency
    @inbounds ζ[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * ζ[i′, Ny, k], ζ[i, Ny, k])

    return nothing
end

@inline function fold_north_face_center_fpivot!(i, k, grid, sign, u)
    Nx, Ny, _ = size(grid)
    i′ = Nx - i + 2 # Element Nx + 1 does not exist?
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy

    for j in 1:Hy
        @inbounds begin
            u[i, Ny + j - 1, k] = sign * u[i′, Ny - j, k]
        end
    end

    return nothing
end

@inline function fold_north_center_face_fpivot!(i, k, grid, sign, v)
    Nx, Ny, _ = size(grid)

    i′ = Nx + 1 - i
    Hy = grid.Hy

    for j in 1:Hy
        @inbounds begin
            v[i, Ny + j, k] = sign * v[i′, Ny - j, k]
        end
    end

    # We substitute the redundant part of the last row of v (index Ny + 1)to ensure consistency
    @inbounds v[i, Ny, k] = ifelse(i > Nx ÷ 2, sign * v[i′, Ny, k], v[i, Ny, k])

    return nothing
end

@inline function fold_north_center_center_fpivot!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)

    i′ = Nx + 1 - i
    Hy = grid.Hy

    for j in 1:Hy
        @inbounds begin
            c[i, Ny - 1 + j, k] = sign * c[i′, Ny - j, k]
        end
    end

    return nothing
end

const CCLocation = Tuple{<:Center, <:Center, <:Any}
const FCLocation = Tuple{<:Face, <:Center, <:Any}
const CFLocation = Tuple{<:Center, <:Face, <:Any}
const FFLocation = Tuple{<:Face, <:Face, <:Any}

# tracers or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::FZBC, ::CCLocation, args...) = fold_north_center_center_fpivot!(i, k, grid, bc.condition, c)

# u-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, u, bc::FZBC, ::FCLocation, args...) = fold_north_face_center_fpivot!(i, k, grid, bc.condition, u)

# v-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, v, bc::FZBC, ::CFLocation, args...) = fold_north_center_face_fpivot!(i, k, grid, bc.condition, v)

# vorticity or similar fields
@inline _fill_north_halo!(i, k, grid, ζ, bc::FZBC, ::FFLocation, args...) = fold_north_face_face_fpivot!(i, k, grid, bc.condition, ζ)

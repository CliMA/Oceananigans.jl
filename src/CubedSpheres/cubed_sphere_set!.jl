using Oceananigans.Fields: AbstractField

import Oceananigans.Fields: set!

function set!(u::ConformalCubedSphereField, v::ConformalCubedSphereField)
    for (u_face, v_face) in zip(u.faces, v.faces)
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

const ConformalCubedSphereFaceFieldᶜᶜᶜ = AbstractField{Center, Center, Center, A, <:ConformalCubedSphereFaceGrid} where A
const ConformalCubedSphereFaceFieldᶠᶜᶜ = AbstractField{Face,   Center, Center, A, <:ConformalCubedSphereFaceGrid} where A
const ConformalCubedSphereFaceFieldᶜᶠᶜ = AbstractField{Center, Face,   Center, A, <:ConformalCubedSphereFaceGrid} where A
const ConformalCubedSphereFaceFieldᶠᶠᶜ = AbstractField{Face,   Face,   Center, A, <:ConformalCubedSphereFaceGrid} where A

const ConformalCubedSphereFaceFieldᶜᶜⁿ = AbstractField{Center, Center, Nothing, A, <:ConformalCubedSphereFaceGrid} where A

function set!(field::ConformalCubedSphereFaceFieldᶜᶜᶜ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        field[i, j, k] = f(grid.λᶜᶜᵃ[i, j], grid.φᶜᶜᵃ[i, j], grid.zᵃᵃᶜ[k])
    end
    return nothing
end

function set!(field::ConformalCubedSphereFaceFieldᶠᶜᶜ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        field[i, j, k] = f(grid.λᶠᶜᵃ[i, j], grid.φᶠᶜᵃ[i, j], grid.zᵃᵃᶜ[k])
    end
    return nothing
end

function set!(field::ConformalCubedSphereFaceFieldᶜᶠᶜ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        field[i, j, k] = f(grid.λᶜᶠᵃ[i, j], grid.φᶜᶠᵃ[i, j], grid.zᵃᵃᶜ[k])
    end
    return nothing
end

function set!(field::ConformalCubedSphereFaceFieldᶠᶠᶜ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        field[i, j, k] = f(grid.λᶠᶠᵃ[i, j], grid.φᶠᶠᵃ[i, j], grid.zᵃᵃᶜ[k])
    end
    return nothing
end

function set!(field::ConformalCubedSphereFaceFieldᶜᶜⁿ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny
        field[i, j, 1] = f(grid.λᶜᶜᵃ[i, j], grid.φᶜᶜᵃ[i, j])
    end
    return nothing
end

set!(field::AbstractCubedSphereField, f::Function) = [set!(field_face, f) for field_face in field.faces]

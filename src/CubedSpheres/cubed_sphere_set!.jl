using Oceananigans.Fields: AbstractField

import Oceananigans.Fields: set!

const ConformalCubedSphereFaceFieldᶜᶜᶜ = AbstractField{Center, Center, Center, A, <:ConformalCubedSphereFaceGrid} where A

function set!(field::ConformalCubedSphereFaceFieldᶜᶜᶜ, f::Function)
    grid = field.grid
    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        field[i, j, k] = f(grid.λᶜᶜᵃ[i, j], grid.φᶜᶜᵃ[i, j], grid.zᵃᵃᶜ[k])
    end
    return nothing
end

set!(field::ConformalCubedSphereField, f::Function) = [set!(field_face, f) for field_face in field.faces]

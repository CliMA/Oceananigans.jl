using Tullio

using Oceananigans.Operators: Δxᶠᶜᵃ, Δyᶜᶠᵃ, Δzᵃᵃᶠ

function accurate_cell_advection_timescale(model::IncompressibleModel)
    grid = model.grid
    Nx, Ny, Nz = size(grid)

    u = view(model.velocities.u.data.parent, 1:Nx, 1:Ny, 1:Nz)
    v = view(model.velocities.v.data.parent, 1:Nx, 1:Ny, 1:Nz)
    w = view(model.velocities.w.data.parent, 1:Nx, 1:Ny, 1:Nz)

    min_timescale = minimum(
        @tullio (min) timescale[k] :=
            (  Δxᶠᶜᵃ(i, j, k, grid) / abs(u[i, j, k])
             + Δyᶜᶠᵃ(i, j, k, grid) / abs(v[i, j, k])
             + Δzᵃᵃᶠ(i, j, k, grid) / abs(w[i, j, k]))
    )

    return min_timescale
end

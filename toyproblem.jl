using Oceananigans
# using GLMakie

Nx = 90
Ny = 40
Nz = 2

# C = Nx = 2πR => 

grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             radius = 2π/Nx,
                             halo = (0, 0, 3),
                             z = (0, 1))

free_surface = ExplicitFreeSurface(gravitational_acceleration=1)
model = HydrostaticFreeSurfaceModel(; grid, tracers=:c, free_surface)

uᵢ(x, y, z) = randn()
vᵢ(x, y, z) = randn()
cᵢ(λ, φ, z) = sin(4π * λ / 360) * exp(-φ^2 / 1800)
set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

u, v, w = model.velocities
c = model.tracers.c

U = 4
Δx = 1 #minimum_xspacing(grid)
Δt = 1e-2 * Δx / U

@show maximum(parent(c))
time_step!(model, Δt)

@show maximum(parent(c))
time_step!(model, Δt)

@show maximum(parent(c))
time_step!(model, Δt)

#=
function nan_halos!(c)
    Nx, Ny, Nz = size(c)
    Hx, Hy, Hz = Oceananigans.Grids.halo_size(c.grid)
    cp = parent(c)
    Tx, Ty, Tz = size(cp)

    view(cp, 1:Hx, :, :) .= NaN
    view(cp, Nx+Hx+1:Tx, :, :) .= NaN
    view(cp, :, 1:Hy, :) .= NaN
    view(cp, :, Ny+Hy+1:Ty, :) .= NaN

    return nothing
end

nan_halos!(c)
=#

#heatmap(parent(c)[:, :, 3])

#=
dx_c = compute!(Field(∂y(c)))
dy_c = compute!(Field(∂x(c)))
dxdy_c = compute!(Field(∂x(∂y(c))))
heatmap(parent(dxdy_c)[:, :, 4])

display(current_figure())
=#

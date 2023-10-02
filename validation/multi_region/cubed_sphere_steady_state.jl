using Printf 
using Oceananigans
using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis

Nx = 30
Ny = 30
Nz = 1

Ω = 1    # sphere's rotation rate
R = 1    # sphere's radius
u₀ = 0.1 # velocity scale
g = 1    # gravitational acceleration

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                z = (-1, 0),
                                radius = R,
                                horizontal_direction_halo = 6,
                                partition = CubedSpherePartition(; R = 1))

#=
latlongrid = LatitudeLongitudeGrid(size=(Nx, Ny, Nz),
                                   longitude = (-90, 90),
                                   latitude = (-45, 45),
                                   z = (-1, 0))
grid = MultiRegionGrid(latlongrid, partition = XPartition(2))
=#

ψᵢ(λ, φ, z) = - u₀ * R * sind(φ)
ηᵢ(λ, φ, z) = u₀^2/(4g) * cosd(2φ) - (2Ω * sind(φ) * u₀ * R)/g * sind(φ)

η = Field{Center, Center, Face}(grid, indices = (:, :, grid.Nz+1))
ψ = Field{Face, Face, Center}(grid)

#=
Here we avoid set! (which also isn't implemented btw) because we would like to manually determine the streamfunction 
within halo regions. This allows us to avoid having to fill_halo_regions correctly for a Face, Face, Center field.
=#

for region in 1:number_of_regions(grid)
    for k in grid.Nz+1, j = 1:grid.Ny, i = 1:grid.Nx
        λ = λnode(i, j, k, grid[region], Center(), Center(), Face())
        φ = φnode(i, j, k, grid[region], Center(), Center(), Face())
        η[region][i, j, k] = ηᵢ(λ, φ, 0)
    end

    for k in 1:grid.Nz, j = 1:grid.Ny+1, i = 1:grid.Nx+1
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵢ(λ, φ, 0)
    end
end

u = XFaceField(grid)
v = YFaceField(grid)

#=
What we want eventually:
u .= - ∂y(ψ)
v .= + ∂x(ψ)
=#

for region in 1:number_of_regions(grid)
    u[region] .= - ∂y(ψ[region])
    v[region] .= + ∂x(ψ[region])
end

closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)
coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    buoyancy = nothing,
                                    coriolis = coriolis,
                                    closure = closure,
                                    tracers = ()
                                    )

set!(model, u = u, v = v, η = η)

# Estimate time-step from the minimum grid spacing.
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)
Δt = 0.2 * min(Δx, Δy) / u₀ # CFL for advection

stop_time = 2π * u₀ / R
simulation = Simulation(model; Δt, stop_time)

# Print a progress message

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n", iteration(sim), prettytime(sim), 
                                prettytime(sim.Δt), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

surface_elevation_field = Field[]
save_surface_elevation(sim) = push!(surface_elevation_field, deepcopy(sim.model.free_surface.η))
simulation.callbacks[:save_surface_elevation] = Callback(save_surface_elevation, IterationInterval(20))

run!(simulation)

@info "Making an animation from the saved data..."

#=
install Imaginocean.jl from GitHub
using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
=# 

#=

using Imaginocean

using GLMakie, GeoMakie

n = Observable(1)

ηₙ = @lift surface_elevation_field[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, ηₙ, colorrange=(-η₀, η₀), colormap = :balance)

fig

frames = 1:length(surface_elevation_field)

GLMakie.record(fig, "cubed_sphere_steady_state.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    ηₙ[] = surface_elevation_field[i]
    heatlatlon!(ax, ηₙ, colorrange=(-η₀, η₀), colormap = :balance)
end

=#
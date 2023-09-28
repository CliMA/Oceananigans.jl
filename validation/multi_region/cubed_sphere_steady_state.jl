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

ψᵢ(λ, φ, z) = - u₀ * R * sind(φ)
ηᵢ(λ, φ, z) = u₀^2/(4g) * cosd(2φ) - (2Ω * sind(φ) * u₀ * R)/g * sind(φ)

η = Field{Center, Center, Center}(grid)
ψ = Field{Face, Face, Center}(grid)

#=
Here we avoid set! (which also isn't implemented btw) because we would like to manually determine the streamfunction 
within halo regions. This allows us to avoid having to fill_halo_regions correctly for a Face, Face, Center field.
=#

for region in 1:number_of_regions(grid)
    
    for k in 1:Nz, j = 1:Ny, i = 1:Nx
        λ = λnode(i, j, k, grid[region], Center(), Center(), Center())
        φ = φnode(i, j, k, grid[region], Center(), Center(), Center())
        η[region][i, j, k] = ηᵢ(λ, φ, 0)
    end  

    for k in 1:Nz, j = 1:Ny+1, i = 1:Nx+1
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
                                    tracers=()
                                    )

#=

# Initial condition for tracer

#=

# 4 Gaussians with width δR (degrees) and magnitude θ₀
δR = 2
θ₀ = 1

# Gaussian 1
i₁ = 1
j₁ = 1
panel = 1
λ₁ = λnode(i₁, j₁, grid[panel], Center(), Center())
φ₁ = φnode(i₁, j₁, grid[panel], Center(), Center())

# Gaussian 2
i₂ = Nx÷4 + 1
j₂ = 3Ny÷4 + 1
panel = 4
λ₂ = λnode(i₂, j₂, grid[panel], Center(), Center())
φ₂ = φnode(i₂, j₂, grid[panel], Center(), Center())

# Gaussian 3
i₃ = 3Nx÷4 + 1
j₃ = 3Ny÷4 + 1
panel = 3
λ₃ = λnode(i₃, j₃, grid[panel], Center(), Center())
φ₃ = φnode(i₃, j₃, grid[panel], Center(), Center())

# Gaussian 4
i₄ = 3Nx÷4+1
j₄ = 3Ny÷4+1
panel = 6
λ₄ = λnode(i₄, j₄, grid[panel], Center(), Center())
φ₄ = φnode(i₄, j₄, grid[panel], Center(), Center())

θᵢ(λ, φ, z) = θ₀ * exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2δR^2) +
              θ₀ * exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2δR^2) +
              θ₀ * exp(-((λ - λ₃)^2 + (φ - φ₃)^2) / 2δR^2) +
              θ₀ * exp(-((λ - λ₄)^2 + (φ - φ₄)^2) / 2δR^2)
=#

θ₀ = 1
Δφ = 20
θᵢ(λ, φ, z) = θ₀ * cosd(4λ) * exp(-φ^2 / 2Δφ^2)

set!(model, θ = θᵢ)

# estimate time-step from the minimum grid spacing
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)
Δt = 0.2 * min(Δx, Δy) / u₀ # CFL for tracer advection

stop_time = 2π * u₀ / R
simulation = Simulation(model; Δt, stop_time)

# Print a progress message

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

tracer_fields = Field[]
save_tracer(sim) = push!(tracer_fields, deepcopy(sim.model.tracers.θ))
simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(20))

run!(simulation)

@info "Making an animation from the saved data..."

# install Imaginocean.jl from GitHub
# using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
using Imaginocean

using GLMakie, GeoMakie

n = Observable(1)

Θₙ = @lift tracer_fields[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "cubed_sphere_tracer_advection.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    Θₙ[] = tracer_fields[i]
    heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)
end

=#
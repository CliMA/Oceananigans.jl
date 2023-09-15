using Oceananigans

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!

using Printf
using GLMakie
GLMakie.activate!()

Nx = 30
Ny = 30
Nz = 1

R = 1 # sphere's radius
U = 1 # velocity scale

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-1, 0),
                                  radius = R,
                                  horizontal_direction_halo = 6,
                                  partition = CubedSpherePartition(; R = 1))


# Solid body rotation
φʳ = 0        # Latitude pierced by the axis of rotation
α  = 90 - φʳ  # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

ψ = Field{Face, Face, Center}(grid)
u = XFaceField(grid)
v = YFaceField(grid)

# Here we avoid set! (which also isn't implemented btw) because we would like
# to manually determine the streamfunction within halo regions. This allows us
# to avoid having to fill_halo_regions correctly for a Face, Face, Center field.
for region in 1:6

    region_grid = getregion(grid, region)

    i₀ = 1
    i⁺ = Nx + 1
    j₀ = 1
    j⁺ = Ny + 1
    k₀ = 1
    k⁺ = Nz + 1

    for k in k₀:k⁺, j=j₀:j⁺, i=i₀:i⁺
        λ = λnode(i, j, k, region_grid, Face(), Face(), Center())
        φ = φnode(i, j, k, region_grid, Face(), Face(), Center())
        getregion(ψ, region)[i, j, k] = ψᵣ(λ, φ, 0)
    end
end

# What we want eventually:
# u .= - ∂y(ψ)
# v .= + ∂x(ψ)

for region in 1:6
    region_ψ = getregion(ψ, region)
    region_u = getregion(u, region)
    region_v = getregion(v, region)

    region_u .= - ∂y(region_ψ)
    region_v .= + ∂x(region_ψ)
end

model = HydrostaticFreeSurfaceModel(; grid,
                                    velocities = PrescribedVelocityFields(; u, v),
                                    momentum_advection = nothing,
                                    tracer_advection = WENO(order=9),
                                    tracers = :θ,
                                    buoyancy = nothing)

# Initial condition for tracer:
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

Δφ = 20
θᵢ(λ, φ, z) = cosd(4λ) * exp(-φ^2 / 2Δφ^2)

set!(model, θ = θᵢ)

# estimate time-step from the minimum grid spacing
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)
Δt = 0.2 * min(Δx, Δy) / U # CFL for tracer advection

stop_time = 2π * U / R
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

n = Observable(1)

Θₙ = []
for region in 1:6
    push!(Θₙ, @lift parent(getregion(tracer_fields[$n], region)[:, :, grid.Nz]))
end

using GeoMakie
using Oceananigans.Utils: get_lat_lon_nodes_and_vertices, get_cartesian_nodes_and_vertices, apply_regionally!

# TODO: import from Imaginocean.jl
function heatlatlon!(ax::Axis, field, k=1; kwargs...)

    LX, LY, LZ = location(field)

    grid = field.grid
    _, (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points = vcat([Point2.(λvertices[:, i, j], φvertices[:, i, j]) for i in axes(λvertices, 2), j in axes(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, kwargs...)

    xlims!(ax, (-180, 180))
    ylims!(ax, (-90, 90))

    return ax
end

heatlatlon!(ax::Axis, field::CubedSphereField, k=1; kwargs...) = apply_regionally!(heatlatlon!, ax, field, k; kwargs...)
heatlatlon!(ax::Axis, field::Observable{<:CubedSphereField}, k=1; kwargs...) = apply_regionally!(heatlatlon!, ax, field.val, k; kwargs...)

Θₙ = @lift tracer_fields[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
#ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
ax = Axis(fig[1, 1])
heatlatlon!(ax, Θₙ, colorrange=(0, 0.5θ₀))

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection_latlon.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    Θₙ[] = tracer_fields[i]
    heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)
end

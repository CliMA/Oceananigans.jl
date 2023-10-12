using Oceananigans, Printf

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!

using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

Nx = 30
Ny = 30
Nz = 1

Lz = 1
R = 1 # sphere's radius
U = 1 # velocity scale
gravitational_acceleration = 100

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-Lz, 0),
                                  radius = R,
                                  horizontal_direction_halo = 4,
                                  partition = CubedSpherePartition(; R = 1))


# Solid body rotation
φʳ = 0        # Latitude pierced by the axis of rotation
α  = 90 - φʳ  # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

ψ = Field{Face, Face, Center}(grid)

# Here we avoid set! (which also isn't implemented btw) because we would like
# to manually determine the streamfunction within halo regions. This allows us
# to avoid having to fill_halo_regions correctly for a Face, Face, Center field.
for region in 1:number_of_regions(grid)
    i₀ = 1
    i⁺ = Nx + 1
    j₀ = 1
    j⁺ = Ny + 1
    k₀ = 1
    k⁺ = Nz + 1

    for k in k₀:k⁺, j=j₀:j⁺, i=i₀:i⁺
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

u = XFaceField(grid)
v = YFaceField(grid)

# What we want eventually:
# u .= - ∂y(ψ)
# v .= + ∂x(ψ)

for region in 1:number_of_regions(grid)
    u[region] .= - ∂y(ψ[region])
    v[region] .= + ∂x(ψ[region])
end

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    # momentum_advection = nothing,
                                    free_surface = ExplicitFreeSurface(; gravitational_acceleration),
                                    tracer_advection = WENO(order=5),
                                    tracers = :θ,
                                    buoyancy = nothing)

# Initial conditions

for region in 1:number_of_regions(grid)
    model.velocities.u[region] .= - ∂y(ψ[region])
    model.velocities.v[region] .= + ∂x(ψ[region])
end

θ₀ = 1
Δφ = 20
θᵢ(λ, φ, z) = θ₀ * cosd(4λ) * exp(-φ^2 / 2Δφ^2)

set!(model, θ = θᵢ)

# Estimate time-step from the minimum grid spacing and the barotropic surface wave speed
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)

c = sqrt(Lz * gravitational_acceleration) # surface wave speed
Δt = 0.1 * min(Δx, Δy) / c # CFL for free surface wave propagation

stop_time = 2π * U / R
stop_time = 100Δt
simulation = Simulation(model; Δt, stop_time)





# Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(5))

tracer_fields = Field[]
save_tracer(sim) = push!(tracer_fields, deepcopy(sim.model.tracers.θ))

ζ = Field{Face, Face, Center}(grid)

vorticity_fields = Field[]

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

function save_vorticity(sim)
    Hx, Hy, Hz = halo_size(grid)

    u, v = sim.model.velocities.u, sim.model.velocities.v

    for passes in 1:3
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
    end
    
    @apply_regionally begin
        params = KernelParameters(size(ζ) .+ 2 .* halo_size(grid), -1 .* halo_size(grid))
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(vorticity_fields, deepcopy(ζ))
end

simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(1))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(1))

run!(simulation)

@info "Making an animation from the saved data..."

# install Imaginocean.jl from GitHub
# using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
using Imaginocean

using GLMakie, GeoMakie

n = Observable(1)

Θₙ = @lift tracer_fields[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
ax = Axis(fig[1, 1])
heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "cubed_sphere_momentum_dynamics.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    Θₙ[] = tracer_fields[i]
    heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)
end





function panel_wise_visualization(field, k=1; hide_decorations = true, colorrange = (-1, 1), colormap = :balance)

    fig = Figure(resolution = (1800, 1200))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    if hide_decorations
        hidedecorations!(ax_1)
        hidedecorations!(ax_2)
        hidedecorations!(ax_3)
        hidedecorations!(ax_4)
        hidedecorations!(ax_5)
        hidedecorations!(ax_6)
    end

    return fig
end

# Now compute vorticity
using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

ζ = Field{Face, Face, Center}(grid)

Hx, Hy, Hz = halo_size(grid)

u, v = model.velocities.u, model.velocities.v

for passes in 1:3
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
end


@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

@apply_regionally begin
    params = KernelParameters(size(ζ) .+ 2 .* halo_size(grid), -1 .* halo_size(grid))
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
end

fig = panel_wise_visualization(ζ, colorrange=(-2, 2))
save("vorticity.png", fig)
fig


n = Observable(1)

ζₙ = @lift vorticity_fields[$n]

k = 1

hide_decorations = true
colorrange = (-1, 1)
colormap = :balance

fig = Figure(resolution = (1800, 1200))

axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
               xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
               xlabel = "Local x direction", ylabel = "Local y direction")

ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
hm_1 = heatmap!(ax_1, parent(getregion(ζₙ[], 1).data[:, :, k]); colorrange, colormap)
Colorbar(fig[3, 2], hm_1)

ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
hm_2 = heatmap!(ax_2, parent(getregion(ζₙ[], 2).data[:, :, k]); colorrange, colormap)
Colorbar(fig[3, 4], hm_2)

ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
hm_3 = heatmap!(ax_3, parent(getregion(ζₙ[], 3).data[:, :, k]); colorrange, colormap)
Colorbar(fig[2, 4], hm_3)

ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
hm_4 = heatmap!(ax_4, parent(getregion(ζₙ[], 4).data[:, :, k]); colorrange, colormap)
Colorbar(fig[2, 6], hm_4)

ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
hm_5 = heatmap!(ax_5, parent(getregion(ζₙ[], 5).data[:, :, k]); colorrange, colormap)
Colorbar(fig[1, 6], hm_5)

ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
hm_6 = heatmap!(ax_6, parent(getregion(ζₙ[], 6).data[:, :, k]); colorrange, colormap)
Colorbar(fig[1, 8], hm_6)

if hide_decorations
    hidedecorations!(ax_1)
    hidedecorations!(ax_2)
    hidedecorations!(ax_3)
    hidedecorations!(ax_4)
    hidedecorations!(ax_5)
    hidedecorations!(ax_6)
end


frames = 1:length(vorticity_fields)

GLMakie.record(fig, "cubed_sphere_momentum_dynamics_vort.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    ζₙ[] = vorticity_fields[i]
end

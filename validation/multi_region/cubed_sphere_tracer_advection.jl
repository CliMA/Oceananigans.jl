using Oceananigans, Printf

using Oceananigans.Grids: node
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!
using JLD2

## Grid setup

Nx, Ny, Nz = 32, 32, 1
R = 1 # sphere's radius
U = 1 # velocity scale

arch = CPU()
grid = ConformalCubedSphereGrid(arch;
                                panel_size = (Nx, Ny, Nz),
                                z = (-1, 0),
                                radius = R,
                                horizontal_direction_halo = 6,
                                partition = CubedSpherePartition(; R = 1))

# Solid body rotation
φʳ = 0       # Latitude pierced by the axis of rotation
α  = 90 - φʳ # Angle between axis of rotation and north pole (degrees)

@inline ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

ψ = Field{Face, Face, Center}(grid)

set!(ψ, ψᵣ)

# Note that set! fills only interior points; to compute u and v, we need information in the halo regions.
fill_halo_regions!(ψ)

# Note that fill_halo_regions! works for (Face, Face, Center) field, *except* for the two corner points that do not
# correspond to an interior point! We need to manually fill the Face-Face halo points of the two corners that do not 
# have a corresponding interior point.

using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: Iterate, launch!

@kernel function _set_ψ_missing_corners!(ψ, grid, region)
    k = @index(Global, Linear)
    i = 1
    j = Ny+1
    if region in (1, 3, 5)
        λ, φ, z = node(i, j, k, grid, Face(), Face(), Center())
        @inbounds ψ[i, j, k] = ψᵣ(λ, φ, z)
    end
    i = Nx+1
    j = 1
    if region in (2, 4, 6)
        λ, φ, z = node(i, j, k, grid, Face(), Face(), Center())
        @inbounds ψ[i, j, k] = ψᵣ(λ, φ, z)
    end
end

region = Iterate(1:6)
@apply_regionally launch!(arch, grid, Nz, _set_ψ_missing_corners!, ψ, grid, region)

u = XFaceField(grid)
v = YFaceField(grid)

@kernel function _set_prescribed_velocities!(ψ, u, v)
    i, j, k = @index(Global, NTuple)
    u[i, j, k] = - ∂y(ψ)[i, j, k]
    v[i, j, k] = + ∂x(ψ)[i, j, k]
end

@apply_regionally launch!(arch, grid, (Nx, Ny, Nz), _set_prescribed_velocities!, ψ, u, v) 

model = HydrostaticFreeSurfaceModel(; grid,
                                    velocities = PrescribedVelocityFields(; u, v),
                                    momentum_advection = nothing,
                                    free_surface = ExplicitFreeSurface(; gravitational_acceleration = 10),
                                    tracer_advection = WENO(order=9),
                                    tracers = :θ,
                                    buoyancy = nothing)

# Initial condition for tracer

#=
using Oceananigans.Grids: λnode, φnode

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

# Estimate time step from the minimum grid spacing based on the CFL condition.
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)
Δt = 0.2 * min(Δx, Δy) / U # CFL for tracer advection

stop_time = 2π * U / R

Ntime = round(Int, stop_time / Δt)

print_output_to_jld2_file = false
if print_output_to_jld2_file
    Ntime = 500
    stop_time = Ntime * Δt
end

@info "Stop time = $(prettytime(stop_time))"
@info "Number of time steps = $Ntime"

simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message_iteration_interval = 10
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|θ|): %.2e, wall time: %s\n", iteration(sim),
                                prettytime(sim), prettytime(sim.Δt), maximum(abs, sim.model.tracers.θ),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(progress_message_iteration_interval))

θ_fields = Field[]
save_θ(sim) = push!(θ_fields, deepcopy(sim.model.tracers.θ))

θ_initial = deepcopy(simulation.model.tracers.θ)

include("cubed_sphere_visualization.jl")

plot_initial_field = false
if plot_initial_field
    # Plot the initial tracer field.
    fig = panel_wise_visualization_with_halos(grid, θ_initial; k = Nz)
    save("cubed_sphere_tracer_advection_θ₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, θ_initial; k = Nz)
    save("cubed_sphere_tracer_advection_θ₀.png", fig)
end

animation_time = 15 # seconds
framerate = 5
n_frames = animation_time * framerate # excluding the initial condition frame
simulation_time_per_frame = stop_time / n_frames
save_fields_iteration_interval = floor(Int, simulation_time_per_frame/Δt)
# Redefine the simulation time per frame.
simulation_time_per_frame = save_fields_iteration_interval * Δt
# Redefine the number of frames.
n_frames = floor(Int, Ntime / save_fields_iteration_interval) # excluding the initial condition frame
# Redefine the animation time.
animation_time = n_frames / framerate
simulation.callbacks[:save_θ] = Callback(save_θ, IterationInterval(save_fields_iteration_interval))

run!(simulation)

if print_output_to_jld2_file
    jldopen("cubed_sphere_tracer_advection_initial_condition.jld2", "w") do file
        for region in 1:6
            file["θ/"*string(region)] = θ_fields[1][region][:, :, Nz]
        end
    end
    jldopen("cubed_sphere_tracer_advection_output.jld2", "w") do file
        for region in 1:6
            file["θ/"*string(region)] = θ_fields[end][region][:, :, Nz]
        end
    end
end

plot_final_field = false
if plot_final_field
    fig = panel_wise_visualization_with_halos(grid, θ_fields[end]; k = Nz)
    save("cubed_sphere_tracer_advection_θ_with_halos.png", fig)

    fig = panel_wise_visualization(grid, θ_fields[end]; k = Nz)
    save("cubed_sphere_tracer_advection_θ.png", fig)
end

θ_colorrange = [-θ₀, θ₀]

plot_snapshots = false
if plot_snapshots
    n_snapshots = 3

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        simulation_time = simulation_time_per_frame * (frame_index - 1)
        title = "Tracer distribution after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, θ_fields[frame_index], title; cbar_label = "tracer level",
                                           specify_plot_limits = true, plot_limits = θ_colorrange)
        save(@sprintf("cubed_sphere_tracer_advection_θ_%d.png", i_snapshot), fig)
    end
end

make_animation = false
if make_animation
    create_panel_wise_visualization_animation(grid, θ_fields, framerate, "cubed_sphere_tracer_advection_θ"; k = Nz)

    prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]

    θ_colorrange = specify_colorrange_timeseries(grid, θ_fields)
    geo_heatlatlon_visualization_animation(grid, θ_fields, "cc", prettytimes, "Tracer distribution"; k = Nz,
                                           cbar_label = "tracer level", specify_plot_limits = true,
                                           plot_limits = θ_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_tracer_advection_θ_geo_heatlatlon_animation")
end

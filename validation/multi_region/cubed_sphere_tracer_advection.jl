using Oceananigans, Printf

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!
using JLD2

include("cubed_sphere_visualization.jl")

Nx, Ny, Nz = 32, 32, 1
R = 1 # sphere's radius
U = 1 # velocity scale

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-1, 0),
                                  radius = R,
                                  horizontal_direction_halo = 6,
                                  partition = CubedSpherePartition(; R = 1))

# Solid body rotation
φʳ = 0       # Latitude pierced by the axis of rotation
α  = 90 - φʳ # Angle between axis of rotation and north pole (degrees)

ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

ψ = Field{Face, Face, Center}(grid)

# Note that set fills only interior points; to compute u and v we need information in the halo regions.
set!(ψ, ψᵣ)

# Note: fill_halo_regions! works for (Face, Face, Center) field, *except* for the two corner points that do not 
# correspond to an interior point! We need to manually fill the Face-Face halo points of the two corners that do not 
# have a corresponding interior point.

for region in [1, 3, 5]
    i = 1
    j = Ny+1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

for region in [2, 4, 6]
    i = Nx+1
    j = 1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

fill_halo_regions!(ψ)

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
                                    velocities = PrescribedVelocityFields(; u, v),
                                    momentum_advection = nothing,
                                    tracer_advection = WENO(order=9),
                                    tracers = :θ,
                                    buoyancy = nothing)

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

# Print a progress message.
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n", iteration(sim), prettytime(sim),
                                prettytime(sim.Δt), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

tracer_fields = Field[]
save_tracer(sim) = push!(tracer_fields, deepcopy(sim.model.tracers.θ))

animation_time = 15 # seconds
framerate = 5
n_frames = animation_time * framerate
simulation_time_per_frame = stop_time / n_frames
# Specify animation_time and framerate in such a way that n_frames is a multiple of n_plots defined below.
save_fields_iteration_interval = floor(Int, simulation_time_per_frame / Δt)
# Redefine the simulation time per frame.
simulation_time_per_frame = save_fields_iteration_interval * Δt
simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(save_fields_iteration_interval))

run!(simulation)

if print_output_to_jld2_file
    jldopen("cubed_sphere_tracer_advection_initial_condition.jld2", "w") do file
        for region in 1:6
            file["tracer/"*string(region)] = tracer_fields[1][region][:, :, Nz]
        end
    end
    jldopen("cubed_sphere_bickley_jet_output.jld2", "w") do file
        for region in 1:6
            file["tracer/"*string(region)] = tracer_fields[end][region][:, :, Nz]
        end
    end
end

n_plots = 3

tracer_colorrange = [-θ₀, θ₀]

for i_plot in 1:n_plots
    frame_index = round(Int, i_plot * n_frames / n_plots)
    simulation_time = simulation_time_per_frame * frame_index
    title = "Tracer distribution after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid, tracer_fields[frame_index], title;
                                       cbar_label = "Tracer level", specify_plot_limits = true,
                                       plot_limits = tracer_colorrange)
    save(@sprintf("tracer_%d.png", i_plot), fig)
end

fig = panel_wise_visualization_with_halos(grid, tracer_fields[end]; k = Nz)
save("tracer_with_halos.png", fig)

fig = panel_wise_visualization(grid, tracer_fields[end]; k = Nz)
save("tracer.png", fig)

create_panel_wise_visualization_animation(grid, tracer_fields, framerate, "tracer"; k = Nz)

prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]
geo_heatlatlon_visualization_animation(grid, tracer_fields, "cc", prettytimes, "Tracer distribution"; k = Nz,
                                       cbar_label = "tracer level", specify_plot_limits = true,
                                       plot_limits = tracer_colorrange, framerate = framerate,
                                       filename = "tracer_geo_heatlatlon_animation")

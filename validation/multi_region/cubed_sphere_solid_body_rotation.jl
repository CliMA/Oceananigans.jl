using Oceananigans, Printf

using Oceananigans.Grids: node, φnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!, Iterate
using Oceananigans.Operators

using JLD2

g = 10

Lz = 1000
R  = 6370e3 # sphere's radius
U  = 40     # velocity scale

# Solid body and planet rotation:
Ω_prime = U/R
π_MITgcm = 3.14159265358979323844
Ω = 2π_MITgcm/86400

## Grid setup

Nx, Ny, Nz = 32, 32, 1
Nhalo = 1
arch = CPU()
grid = ConformalCubedSphereGrid(arch;
                                panel_size = (Nx, Ny, Nz),
                                z = (-Lz, 0),
                                radius = R,
                                horizontal_direction_halo = Nhalo,
                                partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

my_Coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω,
                                           scheme = EnstrophyConserving())

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection = VectorInvariant(vorticity_scheme = EnergyConserving(),
                                                                           vertical_scheme = EnergyConserving()),
                                      free_surface = ExplicitFreeSurface(; gravitational_acceleration = g),
                                      coriolis = my_Coriolis,
                                      buoyancy = nothing)

#=
Specify
model.timestepper.χ = -0.5
to switch to Forward Euler time-stepping with no AB2 step.
=#

# Define the solid body rotation flow field.
@inline ψᵣ(λ, φ, z) = -R^2 * Ω_prime * sind(φ) # ψᵣ(λ, φ, z) = -U * R * sind(φ)

#=
For φʳ = 90:
ψᵣ(λ, φ, z) = - U * R * sind(φ)
uᵣ(λ, φ, z) = - 1 / R * ∂φ(ψᵣ) = U * cosd(φ)
vᵣ(λ, φ, z) = + 1 / (R * cosd(φ)) * ∂λ(ψᵣ) = 0
ζᵣ(λ, φ, z) = - 1 / (R * cosd(φ)) * ∂φ(uᵣ * cosd(φ)) = 2 * (U / R) * sind(φ)
=#
ψ = Field{Face, Face, Center}(grid)

set!(ψ, ψᵣ)

# Note that set! fills only interior points; to compute u and v, we need information in the halo regions.
fill_halo_regions!(ψ)

# Note that fill_halo_regions! works for (Face, Face, Center) field, *except* for the two corner points that do not
# correspond to an interior point! We need to manually fill the Face-Face halo points of the two corners that do not
# have a corresponding interior point.

using KernelAbstractions: @kernel, @index
using Oceananigans.Utils

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

@kernel function _set_initial_velocities!(ψ, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = - ∂y(ψ)[i, j, k]
    @inbounds v[i, j, k] = + ∂x(ψ)[i, j, k]
end

@apply_regionally launch!(arch, grid, (Nx, Ny, Nz), _set_initial_velocities!, ψ, model.velocities.u, model.velocities.v)

fill_halo_regions!((model.velocities.u, model.velocities.v))

# Set the initial conditions.
fac = -(R^2) * Ω_prime * (Ω + 0.5Ω_prime) / g

@kernel function _set_initial_surface_elevation!(model_η, grid)
    k = Nz+1
    i, j = @index(Global, NTuple)
    φ = φnode(i, j, k, grid, Center(), Center(), Face())
    @inbounds model_η[i, j, k] = fac * ((sind(φ))^2 - 1/3)
end

@apply_regionally launch!(model.architecture, grid, (Nx, Ny), _set_initial_surface_elevation!, model.free_surface.η,
                          grid)

fill_halo_regions!(model.free_surface.η)

Δt = 600
stop_time = 10*86400 # 10 days, close to revolution period = 11.58 days

Ntime = round(Int, stop_time/Δt)

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
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n", iteration(sim),
                                prettytime(sim), prettytime(sim.Δt), maximum(abs, sim.model.velocities.u),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(progress_message_iteration_interval))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

# Now, compute the vorticity.
ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

offset = -1 .* halo_size(grid)

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, model.velocities.u, model.velocities.v)
end

ζ_fields = Field[]

function save_ζ(sim)
    grid = sim.model.grid
    
    offset = -1 .* halo_size(grid)
    
    u, v, _ = sim.model.velocities

    fill_halo_regions!((u, v))

    @apply_regionally begin
        params = KernelParameters(total_size(ζ[1]), offset)
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(ζ_fields, deepcopy(ζ))
end

η_fields = Field[]
save_η(sim) = push!(η_fields, deepcopy(sim.model.free_surface.η))

u₀ = deepcopy(simulation.model.velocities.u)
v₀ = deepcopy(simulation.model.velocities.v)
ζ₀ = deepcopy(ζ)

η₀ = deepcopy(simulation.model.free_surface.η)
# Redefine η₀ as η₀ = η₀ - Lz for better visualization.
for region in 1:number_of_regions(grid)
    for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
        η₀[region][i, j, k] -= Lz
    end
end

include("cubed_sphere_visualization.jl")

plot_initial_field = false
if plot_initial_field
    # Plot the initial velocity field.
    fig = panel_wise_visualization_with_halos(grid, u₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_u₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_u₀.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_v₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_v₀.png", fig)

    # Plot the initial vorticity field.
    fig = panel_wise_visualization_with_halos(grid, ζ₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_ζ₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ₀; k = Nz)
    save("cubed_sphere_solid_body_rotation_ζ₀.png", fig)

    # Plot the initial surface elevation field.
    fig = panel_wise_visualization_with_halos(grid, η₀; k = Nz + 1, ssh = true)
    save("cubed_sphere_solid_body_rotation_η₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, η₀; k = Nz + 1, ssh = true)
    save("cubed_sphere_solid_body_rotation_η₀.png", fig)
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
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_ζ] = Callback(save_ζ, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_η] = Callback(save_η, IterationInterval(save_fields_iteration_interval))

run!(simulation)

if print_output_to_jld2_file
    jldopen("cubed_sphere_solid_body_rotation_initial_condition.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[1][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[1][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[1][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[1][region][:, :, Nz+1]
        end
    end
    jldopen("cubed_sphere_solid_body_rotation_output.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[end][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[end][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[end][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[end][region][:, :, Nz+1]
        end
    end
end

# Redefine η as η = η - Lz for better visualization.
for i_frame in 1:n_frames+1
    for region in 1:number_of_regions(grid)
        for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
            η_fields[i_frame][region][i, j, k] -= Lz
        end
    end
end

plot_final_field = false
if plot_final_field
    fig = panel_wise_visualization_with_halos(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_u_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_u.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_v_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_v.png", fig)

    fig = panel_wise_visualization_with_halos(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_ζ_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_solid_body_rotation_ζ.png", fig)

    fig = panel_wise_visualization_with_halos(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_solid_body_rotation_η_with_halos.png", fig)

    fig = panel_wise_visualization(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_solid_body_rotation_η.png", fig)
end

plot_snapshots = false
if plot_snapshots
    n_snapshots = 3

    u_colorrange = zeros(2)
    v_colorrange = zeros(2)
    ζ_colorrange = zeros(2)
    η_colorrange = zeros(2)

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        u_colorrange_at_frame_index = specify_colorrange(grid, u_fields[frame_index])
        v_colorrange_at_frame_index = specify_colorrange(grid, v_fields[frame_index])
        ζ_colorrange_at_frame_index = specify_colorrange(grid, ζ_fields[frame_index])
        η_colorrange_at_frame_index = specify_colorrange(grid, η_fields[frame_index]; ssh = true)
        if i_snapshot == 0
            u_colorrange[:] = collect(u_colorrange_at_frame_index)
            v_colorrange[:] = collect(v_colorrange_at_frame_index)
            ζ_colorrange[:] = collect(ζ_colorrange_at_frame_index)
            η_colorrange[:] = collect(η_colorrange_at_frame_index)
        else
            u_colorrange[1] = min(u_colorrange[1], u_colorrange_at_frame_index[1])
            u_colorrange[2] = max(u_colorrange[2], u_colorrange_at_frame_index[2])
            v_colorrange[1] = min(v_colorrange[1], v_colorrange_at_frame_index[1])
            v_colorrange[2] = max(v_colorrange[2], v_colorrange_at_frame_index[2])
            ζ_colorrange[1] = min(ζ_colorrange[1], ζ_colorrange_at_frame_index[1])
            ζ_colorrange[2] = max(ζ_colorrange[2], ζ_colorrange_at_frame_index[2])
            η_colorrange[1] = min(η_colorrange[1], η_colorrange_at_frame_index[1])
            η_colorrange[2] = max(η_colorrange[2], η_colorrange_at_frame_index[2])
        end
    end

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        simulation_time = simulation_time_per_frame * (frame_index - 1)
        #=
        title = "Zonal velocity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, u_fields[frame_index],
                                                                                          "fc"), title;
                                           cbar_label = "zonal velocity", specify_plot_limits = true,
                                           plot_limits = u_colorrange)
        save(@sprintf("cubed_sphere_solid_body_rotation_u_%d.png", i_snapshot), fig)
        title = "Meridional velocity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, v_fields[frame_index],
                                                                                          "cf"), title;
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange)
        save(@sprintf("cubed_sphere_solid_body_rotation_v_%d.png", i_snapshot), fig)
        =#
        title = "Relative vorticity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[frame_index],
                                                                                          "ff"), title;
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange)
        save(@sprintf("cubed_sphere_solid_body_rotation_ζ_%d.png", i_snapshot), fig)
        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; ssh = true,
                                           cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_solid_body_rotation_η_%d.png", i_snapshot), fig)
    end
end

make_animations = false
if make_animations
    create_panel_wise_visualization_animation(grid, u_fields, framerate, "cubed_sphere_solid_body_rotation_u"; k = Nz)
    create_panel_wise_visualization_animation(grid, v_fields, framerate, "cubed_sphere_solid_body_rotation_v"; k = Nz)
    create_panel_wise_visualization_animation(grid, ζ_fields, framerate, "cubed_sphere_solid_body_rotation_ζ"; k = Nz)
    create_panel_wise_visualization_animation(grid, η_fields, framerate, "cubed_sphere_solid_body_rotation_η"; k = Nz+1,
                                              ssh = true)

    prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]

    u_colorrange = specify_colorrange_timeseries(grid, u_fields)
    geo_heatlatlon_visualization_animation(grid, u_fields, "fc", prettytimes, "Zonal velocity"; k = Nz,
                                           cbar_label = "zonal velocity", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_solid_body_rotation_u_geo_heatlatlon_animation")

    v_colorrange = specify_colorrange_timeseries(grid, v_fields)
    geo_heatlatlon_visualization_animation(grid, v_fields, "cf", prettytimes, "Meridional velocity"; k = Nz,
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_solid_body_rotation_v_geo_heatlatlon_animation")

    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_fields)
    geo_heatlatlon_visualization_animation(grid, ζ_fields, "ff", prettytimes, "Relative vorticity"; k = Nz,
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_solid_body_rotation_ζ_geo_heatlatlon_animation")

    #=
    η_colorrange = specify_colorrange_timeseries(grid, η_fields; ssh = true)
    geo_heatlatlon_visualization_animation(grid, η_fields, "cc", prettytimes, "Surface elevation"; k = Nz+1,
                                           ssh = true, cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_solid_body_rotation_η_geo_heatlatlon_animation")
    =#
end

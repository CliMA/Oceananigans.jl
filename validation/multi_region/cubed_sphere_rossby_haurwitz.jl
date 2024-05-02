using Oceananigans, Printf

using Oceananigans.Grids: λnode, φnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.Utils: Iterate

using JLD2

## Grid setup

R = 6371e3
H = 8000

Nx, Ny, Nz = 32, 32, 1
Nhalo = 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-H, 0),
                                  radius = R,
                                  horizontal_direction_halo = Nhalo,
                                  partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

## Model setup

horizontal_closure = nothing

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection = nothing,
                                      free_surface = ExplicitFreeSurface(; gravitational_acceleration = 100),
                                      coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving()),
                                      closure = (horizontal_closure),
                                      tracers = nothing,
                                      buoyancy = nothing)

## Rossby-Haurwitz initial condition from Williamson et al. (§3.6, 1992)
## # Here: θ ∈ [-π/2, π/2] is latitude and ϕ ∈ [0, 2π) is longitude.

K = 7.848e-6
ω = 0
n = 4

g = model.free_surface.gravitational_acceleration
Ω = model.coriolis.rotation_rate

A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2)
B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * (n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # Why not (n+1)^2 sin(θ)^2 + 1?
C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ((n+1) * cos(θ)^2 - (n+2))

ψ_function(θ, ϕ) = -R^2 * ω * sin(θ) + R^2 * K * cos(θ)^n * sin(θ) * cos(n*ϕ)

u_function(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ)
v_function(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ)

h_function(θ, ϕ) = H + R^2/g * (A(θ) + B(θ) * cos(n * ϕ) + C(θ) * cos(2n * ϕ))

# Initial conditions
# Previously: θ ∈ [-π/2, π/2] is latitude and ϕ ∈ [0, 2π) is longitude
# Oceananigans: ϕ ∈ [-90, 90] and λ ∈ [-180, 180]

rescale¹(λ) = (λ + 180) / 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ

# Arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
#=
u₀(λ, ϕ, z) = u_function(rescale²(ϕ), rescale¹(λ))
v₀(λ, ϕ, z) = v_function(rescale²(ϕ), rescale¹(λ))
=#
η₀(λ, ϕ)    = h_function(rescale²(ϕ), rescale¹(λ))

#=
set!(model, u=u₀, v=v₀, η = η₀)
=#

ψ₀(λ, φ, z) = ψ_function(rescale²(φ), rescale¹(λ))

ψ = Field{Face, Face, Center}(grid)

# Note that set! fills only interior points; to compute u and v we need information in the halo regions.
set!(ψ, ψ₀)

for region in [1, 3, 5]
    i = 1
    j = Ny+1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψ₀(λ, φ, 0)
    end
end

for region in [2, 4, 6]
    i = Nx+1
    j = 1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψ₀(λ, φ, 0)
    end
end

fill_halo_regions!(ψ)

u = XFaceField(grid)
v = YFaceField(grid)

for region in 1:number_of_regions(grid)
    for j in 1:Ny, i in 1:Nx, k in 1:Nz
        u[region][i, j, k] = - (ψ[region][i, j+1, k] - ψ[region][i, j, k]) / grid[region].Δyᶠᶜᵃ[i, j]
        v[region][i, j, k] =   (ψ[region][i+1, j, k] - ψ[region][i, j, k]) / grid[region].Δxᶜᶠᵃ[i, j]
    end
end

fill_halo_regions!((u, v))

# Now, compute the vorticity.
using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

offset = -1 .* halo_size(grid)

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
end

for region in 1:number_of_regions(grid)
    for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in 1:Nz
        model.velocities.u[region][i,j,k] = u[region][i, j, k]
        model.velocities.v[region][i,j,k] = v[region][i, j, k]
    end
    
    for j in 1:Ny, i in 1:Nx, k in Nz+1:Nz+1
        λ = λnode(i, j, k, grid[region], Center(), Center(), Face())
        φ = φnode(i, j, k, grid[region], Center(), Center(), Face())
        model.free_surface.η[region][i, j, k] = η₀(λ, φ)
    end
end

fill_halo_regions!(model.free_surface.η)

## Simulation setup

# Compute amount of time needed for a 45° rotation.
angular_velocity = (n * (3+n) * ω - 2Ω) / ((1+n) * (2+n))
stop_time = deg2rad(360) / abs(angular_velocity)

min_spacing = filter(!iszero, grid[1].Δxᶠᶠᵃ) |> minimum
c = sqrt(model.free_surface.gravitational_acceleration * H)
Δt = 0.2 * min_spacing / c

Ntime = round(Int, stop_time/Δt)

print_output_to_jld2_file = false
if print_output_to_jld2_file
    Ntime = 500
    stop_time = Ntime * Δt
end

@info "Stop time = $(prettytime(stop_time))"
@info "Number of time steps = $Ntime"

gravity_wave_speed = sqrt(g * H)
min_spacing = filter(!iszero, grid[1].Δyᶠᶠᵃ) |> minimum
wave_propagation_time_scale = min_spacing / gravity_wave_speed
gravity_wave_cfl = Δt / wave_propagation_time_scale
@info "Gravity wave CFL = $gravity_wave_cfl"

if !isnothing(model.closure)
    ν = model.closure.ν
    diffusive_cfl = ν * Δt / min_spacing^2
    @info "Diffusive CFL = $diffusive_cfl"
end

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

ζ = Field{Face, Face, Center}(grid)

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
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

uᵢ = deepcopy(simulation.model.velocities.u)
vᵢ = deepcopy(simulation.model.velocities.v)
ζᵢ = deepcopy(ζ) 

ηᵢ = deepcopy(simulation.model.free_surface.η)
ηᵢ_mean = 0.5 * (maximum(ηᵢ) + minimum(ηᵢ))
# Redefine ηᵢ as ηᵢ = ηᵢ - ηᵢ_mean for better visualization.
for region in 1:number_of_regions(grid)
    for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
        ηᵢ[region][i, j, k] -= ηᵢ_mean
    end
end

include("cubed_sphere_visualization.jl")

plot_initial_field = false
if plot_initial_field
    # Plot the initial velocity field.
    fig = panel_wise_visualization_with_halos(grid, uᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_u₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, uᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_u₀.png", fig)

    fig = panel_wise_visualization_with_halos(grid, vᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_v₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, vᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_v₀.png", fig)

    # Plot the initial vorticity field.
    fig = panel_wise_visualization_with_halos(grid, ζᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_ζ₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζᵢ; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_ζ₀.png", fig)

    # Plot the initial surface elevation field.
    fig = panel_wise_visualization_with_halos(grid, ηᵢ; k = Nz + 1, ssh = true)
    save("cubed_sphere_rossby_haurwitz_wave_η₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ηᵢ; k = Nz + 1, ssh = true)
    save("cubed_sphere_rossby_haurwitz_wave_η₀.png", fig)
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
    jldopen("cubed_sphere_rossby_haurwitz_initial_condition.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[1][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[1][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[1][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[1][region][:, :, Nz+1]
        end
    end
    jldopen("cubed_sphere_rossby_haurwitz_output.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[end][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[end][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[end][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[end][region][:, :, Nz+1]
        end
    end
end

# Redefine η as η = η - ηᵢ_mean for better visualization.
for i_frame in 1:n_frames+1
    for region in 1:number_of_regions(grid)
        for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
            η_fields[i_frame][region][i, j, k] -= ηᵢ_mean
        end
    end
end

plot_final_field = false
if plot_final_field
    fig = panel_wise_visualization_with_halos(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_u_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_u.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_v_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_v.png", fig)

    fig = panel_wise_visualization_with_halos(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_ζ_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_rossby_haurwitz_wave_ζ.png", fig)

    fig = panel_wise_visualization_with_halos(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_rossby_haurwitz_wave_η_with_halos.png", fig)

    fig = panel_wise_visualization(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_rossby_haurwitz_wave_η.png", fig)
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
        save(@sprintf("cubed_sphere_rossby_haurwitz_wave_u_%d.png", i_snapshot), fig)
        title = "Meridional velocity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, v_fields[frame_index],
                                                                                          "cf"), title;
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange)
        save(@sprintf("cubed_sphere_rossby_haurwitz_wave_v_%d.png", i_snapshot), fig)
        =#
        title = "Relative vorticity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[frame_index],
                                                                                          "ff"), title;
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange)
        save(@sprintf("cubed_sphere_rossby_haurwitz_wave_ζ_%d.png", i_snapshot), fig)
        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; ssh = true,
                                           cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_rossby_haurwitz_wave_η_%d.png", i_snapshot), fig)
    end
end

make_animations = false
if make_animations
    create_panel_wise_visualization_animation(grid, u_fields, framerate, "cubed_sphere_rossby_haurwitz_wave_u"; k = Nz)
    create_panel_wise_visualization_animation(grid, v_fields, framerate, "cubed_sphere_rossby_haurwitz_wave_v"; k = Nz)
    create_panel_wise_visualization_animation(grid, ζ_fields, framerate, "cubed_sphere_rossby_haurwitz_wave_ζ"; k = Nz)
    create_panel_wise_visualization_animation(grid, η_fields, framerate, "cubed_sphere_rossby_haurwitz_wave_η";
                                              k = Nz+1, ssh = true)

    prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]

    u_colorrange = specify_colorrange_timeseries(grid, u_fields)
    geo_heatlatlon_visualization_animation(grid, u_fields, "fc", prettytimes, "Zonal velocity"; k = Nz,
                                           cbar_label = "zonal velocity", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_rossby_haurwitz_wave_u_geo_heatlatlon_animation")

    v_colorrange = specify_colorrange_timeseries(grid, v_fields)
    geo_heatlatlon_visualization_animation(grid, v_fields, "cf", prettytimes, "Meridional velocity"; k = Nz,
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_rossby_haurwitz_wave_v_geo_heatlatlon_animation")

    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_fields)
    geo_heatlatlon_visualization_animation(grid, ζ_fields, "ff", prettytimes, "Relative vorticity"; k = Nz,
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_rossby_haurwitz_wave_ζ_geo_heatlatlon_animation")

    #=
    η_colorrange = specify_colorrange_timeseries(grid, η_fields; ssh = true)
    geo_heatlatlon_visualization_animation(grid, η_fields, "cc", prettytimes, "Surface elevation"; k = Nz+1,
                                           ssh = true, cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_rossby_haurwitz_wave_η_geo_heatlatlon_animation")
    =#
end

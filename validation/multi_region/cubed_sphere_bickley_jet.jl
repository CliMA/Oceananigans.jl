using Oceananigans, Printf

using Oceananigans.Grids: node, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!, Iterate
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils

using JLD2

# Definition of the "Bickley jet": a sech(y)^2 jet with sinusoidal tracer
@inline Ψ(y) = - tanh(y)
@inline U(y) = sech(y)^2

# A sinusoidal tracer
@inline C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
@inline ψ̃(x, y, ℓ, k_x, k_y) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k_x * x) * cos(k_y * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
@inline ũ(x, y, ℓ, k_x, k_y) = +ψ̃(x, y, ℓ, k_x, k_y) * (k_y * tan(k_y * y) + (y + ℓ / 10) / ℓ^2)
@inline ṽ(x, y, ℓ, k_x, k_y) = -ψ̃(x, y, ℓ, k_x, k_y) * k_x * tan(k_x * x)

"""
    set_bickley_jet!(model; Ly = 4π, ϵ = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

Set the `u` and `v`: Large-scale jet + vortical perturbations.
Set the tracer `c`: Sinusoid

Keyword args
============

* `Ly`: meridional domain extent
* `ϵ`: perturbation magnitude
* `ℓ₀`: Gaussian width for meridional extent of 4π
* `k₀`: sinusoidal wavenumber for domain extents of 4π in each direction
"""
function set_bickley_jet!(model; Lx = 4π, Ly = 4π, ϵ = 0.1, ℓ₀ = 0.5, k₀ = 0.5)
    ℓ = ℓ₀ / 4π * Ly
    k_x = k₀ * 4π / Lx
    k_y = k₀ * 4π / Ly

    dr(x) = deg2rad(x)

    @inline ψᵢ(λ, φ, z) = Ψ(dr(φ) * 8) + ϵ * ψ̃(dr(λ) * 2, dr(φ) * 8, ℓ, k_x, k_y)
    @inline cᵢ(λ, φ, z) = C(dr(φ)*8, 180)

    grid = model.grid
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    ψ = Field{Face, Face, Center}(grid)

    set!(ψ, ψᵢ)

    # Note that set! fills only interior points; to compute u and v, we need information in the halo regions.
    fill_halo_regions!(ψ)

    # Note that fill_halo_regions! works for (Face, Face, Center) field, *except* for the two corner points that do not
    # correspond to an interior point! We need to manually fill the Face-Face halo points of the two corners that do not
    # have a corresponding interior point.

    @kernel function _set_ψ_missing_corners!(ψ, grid, region)
        k = @index(Global, Linear)
        i = 1
        j = Ny+1
        if region in (1, 3, 5)
            λ, φ, z = node(i, j, k, grid, Face(), Face(), Center())
            @inbounds ψ[i, j, k] = ψᵢ(λ, φ, z)
        end
        i = Nx+1
        j = 1
        if region in (2, 4, 6)
            λ, φ, z = node(i, j, k, grid, Face(), Face(), Center())
            @inbounds ψ[i, j, k] = ψᵢ(λ, φ, z)
        end
    end

    region = Iterate(1:6)
    @apply_regionally launch!(arch, grid, Nz, _set_ψ_missing_corners!, ψ, grid, region)

    @kernel function _set_initial_velocities!(ψ, u, v)
        i, j, k = @index(Global, NTuple)
        @inbounds u[i, j, k] = - ∂y(ψ)[i, j, k]
        @inbounds v[i, j, k] = + ∂x(ψ)[i, j, k]
    end

    @apply_regionally launch!(arch, grid, (Nx, Ny, Nz), _set_initial_velocities!, ψ, model.velocities.u,
                              model.velocities.v)

    u_v_max = max(maximum(abs, model.velocities.u), maximum(abs, model.velocities.v))

    @kernel function _normalize_initial_velocities!(u, v)
        i, j, k = @index(Global, NTuple)
        @inbounds u[i, j, k] /= u_v_max
        @inbounds v[i, j, k] /= u_v_max
    end

    @apply_regionally launch!(arch, grid, (Nx, Ny, Nz), _normalize_initial_velocities!, model.velocities.u,
                              model.velocities.v)

    fill_halo_regions!((model.velocities.u, model.velocities.v))

    @kernel function _set_initial_tracer_distribution!(c, grid)
        i, j, k = @index(Global, NTuple)
        λ, φ, z = node(i, j, k, grid, Center(), Center(), Center())
        @inbounds c[i, j, k] = cᵢ(λ, φ, z)
    end

    @apply_regionally launch!(model.architecture, grid, (Nx, Ny, Nz), _set_initial_tracer_distribution!,
                              model.tracers.c, grid)

    fill_halo_regions!(model.tracers.c)

    return nothing
end

## Grid setup

H = 1

Nx, Ny, Nz = 32, 32, 1
Nhalo = 4
R = 1 # sphere's radius

arch = CPU()
grid = ConformalCubedSphereGrid(arch;
                                panel_size = (Nx, Ny, Nz),
                                z = (-H, 0),
                                radius = R,
                                horizontal_direction_halo = Nhalo,
                                partition = CubedSpherePartition(; R = 1))

Lx = 2π
Ly = π

Hx, Hy, Hz = halo_size(grid)

momentum_advection = VectorInvariant()
tracer_advection = WENO()
g = 1
free_surface = ExplicitFreeSurface(gravitational_acceleration=g)

c = sqrt(g * H)
closure = ScalarDiffusivity(ν = c * R * 0.0005 * 32 / Nx)
#=
Using N₁ = 32, r₁ = 1, g₁ = 1, H₁ = 1, so that c₁ = sqrt(g₁ * H₁) = 1, and
Δt₁ = 0.1 * min_spacing / c₁ = 0.1 * min_spacing, the numerical solution becomes unstable at ν = 0.0001, and experiences
excessive diffusion at ν₁ = 0.001. So, we specify ν₁ = 0.0005, a value midway between 0.0001 and 0.001, to balance
between retaining important gradient features and maintaining stability. For a much larger radius of r₂ = 6370e3
(Earth's radius) at same spatial resolution (N₂ = 32), we apply scaling analysis and obtain
ν₂ = c₂/c₁ (r₂/N₂) / (r₁/N₁) ν₁ = c₂/c₁ r₂/r₁ ν₁ = c₂ r₂ ν₁ = 100 * 6370e3 * 0.0005.
=#

Ω = 2π/86400 # Earth's rotation rate
myCoriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω, scheme = EnstrophyConserving())

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection,
                                      tracer_advection,
                                      free_surface,
                                      tracers = :c,
                                      coriolis = myCoriolis,
                                      buoyancy = nothing,
                                      closure = closure)

set_bickley_jet!(model; Lx = Lx, Ly = Ly, ϵ = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

# Specify cfl = c * Δt / min(Δx) = 0.2, so that Δt = 0.2 * min(Δx) / c
min_spacing = filter(!iszero, grid[1].Δxᶠᶠᵃ) |> minimum
Δt = 0.2 * min_spacing / c

Ntime = 15000
stop_time = Ntime * Δt

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
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f, max|c|: %.3f, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u),
                                maximum(abs, model.free_surface.η), maximum(abs, model.tracers.c),
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
    kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
    launch!(arch, grid, kernel_parameters, _compute_vorticity!, ζ, grid, model.velocities.u, model.velocities.v)
end

ζ_fields = Field[]

function save_ζ(sim)
    grid = sim.model.grid
    
    offset = -1 .* halo_size(grid)
    
    u, v, _ = sim.model.velocities

    fill_halo_regions!((u, v))

    @apply_regionally begin
        kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
        launch!(arch, grid, kernel_parameters, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(ζ_fields, deepcopy(ζ))
end

η_fields = Field[]
save_η(sim) = push!(η_fields, deepcopy(sim.model.free_surface.η))

c_fields = Field[]
save_c(sim) = push!(c_fields, deepcopy(sim.model.tracers.c))

uᵢ = deepcopy(simulation.model.velocities.u)
vᵢ = deepcopy(simulation.model.velocities.v)
ζᵢ = deepcopy(ζ) 

ηᵢ = deepcopy(simulation.model.free_surface.η)
# Redefine ηᵢ as ηᵢ = ηᵢ - H for better visualization.
for region in 1:number_of_regions(grid)
    for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
        ηᵢ[region][i, j, k] -= H
    end
end

cᵢ = deepcopy(simulation.model.tracers.c)

include("cubed_sphere_visualization.jl")

plot_initial_field = false
if plot_initial_field
    # Plot the initial velocity field.
    fig = panel_wise_visualization_with_halos(grid, uᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_u₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, uᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_u₀.png", fig)

    fig = panel_wise_visualization_with_halos(grid, vᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_v₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, vᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_v₀.png", fig)

    # Plot the initial vorticity field.
    fig = panel_wise_visualization_with_halos(grid, ζᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_ζ₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_ζ₀.png", fig)

    # Plot the initial tracer field.
    fig = panel_wise_visualization_with_halos(grid, cᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_c₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, cᵢ; k = Nz)
    save("cubed_sphere_bickley_jet_c₀.png", fig)
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
simulation.callbacks[:save_c] = Callback(save_c, IterationInterval(save_fields_iteration_interval))

run!(simulation)

if print_output_to_jld2_file
    jldopen("cubed_sphere_bickley_jet_initial_condition.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[1][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[1][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[1][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[1][region][:, :, Nz+1]
            file["c/"*string(region)] = c_fields[1][region][:, :, Nz]
        end
    end
    jldopen("cubed_sphere_bickley_jet_output.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[end][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[end][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[end][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[end][region][:, :, Nz+1]
            file["c/"*string(region)] = c_fields[end][region][:, :, Nz]
        end
    end
end

# Redefine η as η = η - H for better visualization.
for i_frame in 1:n_frames+1
    for region in 1:number_of_regions(grid)
        for j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx, k in Nz+1:Nz+1
            η_fields[i_frame][region][i, j, k] -= H
        end
    end
end

plot_final_field = false
if plot_final_field
    fig = panel_wise_visualization_with_halos(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_u_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_u.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_v_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_v.png", fig)

    fig = panel_wise_visualization_with_halos(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_ζ_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_ζ.png", fig)

    fig = panel_wise_visualization_with_halos(grid, η_fields[end]; k = Nz + 1, use_symmetric_colorrange = false,
                                              ssh = true)
    save("cubed_sphere_bickley_jet_η_with_halos.png", fig)

    fig = panel_wise_visualization(grid, η_fields[end]; k = Nz + 1, use_symmetric_colorrange = false, ssh = true)
    save("cubed_sphere_bickley_jet_η.png", fig)

    fig = panel_wise_visualization_with_halos(grid, c_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_c_with_halos.png", fig)

    fig = panel_wise_visualization(grid, c_fields[end]; k = Nz)
    save("cubed_sphere_bickley_jet_c.png", fig)
end

plot_snapshots = false
if plot_snapshots
    n_snapshots = 3

    u_colorrange = zeros(2)
    v_colorrange = zeros(2)
    ζ_colorrange = zeros(2)
    η_colorrange = zeros(2)
    c_colorrange = zeros(2)

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        u_colorrange_at_frame_index = specify_colorrange(grid, u_fields[frame_index])
        v_colorrange_at_frame_index = specify_colorrange(grid, v_fields[frame_index])
        ζ_colorrange_at_frame_index = specify_colorrange(grid, ζ_fields[frame_index])
        η_colorrange_at_frame_index = specify_colorrange(grid, η_fields[frame_index]; use_symmetric_colorrange = false,
                                                         ssh = true)
        c_colorrange_at_frame_index = specify_colorrange(grid, c_fields[frame_index])
        if i_snapshot == 0
            u_colorrange[:] = collect(u_colorrange_at_frame_index)
            v_colorrange[:] = collect(v_colorrange_at_frame_index)
            ζ_colorrange[:] = collect(ζ_colorrange_at_frame_index)
            η_colorrange[:] = collect(η_colorrange_at_frame_index)
            c_colorrange[:] = collect(c_colorrange_at_frame_index)
        else
            u_colorrange[1] = min(u_colorrange[1], u_colorrange_at_frame_index[1])
            u_colorrange[2] = max(u_colorrange[2], u_colorrange_at_frame_index[2])
            v_colorrange[1] = min(v_colorrange[1], v_colorrange_at_frame_index[1])
            v_colorrange[2] = max(v_colorrange[2], v_colorrange_at_frame_index[2])
            ζ_colorrange[1] = min(ζ_colorrange[1], ζ_colorrange_at_frame_index[1])
            ζ_colorrange[2] = max(ζ_colorrange[2], ζ_colorrange_at_frame_index[2])
            η_colorrange[1] = min(η_colorrange[1], η_colorrange_at_frame_index[1])
            η_colorrange[2] = max(η_colorrange[2], η_colorrange_at_frame_index[2])
            c_colorrange[1] = min(c_colorrange[1], c_colorrange_at_frame_index[1])
            c_colorrange[2] = max(c_colorrange[2], c_colorrange_at_frame_index[2])
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
        save(@sprintf("cubed_sphere_bickley_jet_u_%d.png", i_snapshot), fig)
        title = "Meridional velocity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, v_fields[frame_index],
                                                                                          "cf"), title;
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange)
        save(@sprintf("cubed_sphere_bickley_jet_v_%d.png", i_snapshot), fig)
        =#
        title = "Relative vorticity after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid,
                                           interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[frame_index],
                                                                                          "ff"), title;
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange)
        save(@sprintf("cubed_sphere_bickley_jet_ζ_%d.png", i_snapshot), fig)
        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; use_symmetric_colorrange = false,
                                           ssh = true, cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_bickley_jet_η_%d.png", i_snapshot), fig)
        title = "Tracer distribution after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, c_fields[frame_index], title; cbar_label = "tracer level",
                                           specify_plot_limits = true, plot_limits = c_colorrange)
        save(@sprintf("cubed_sphere_bickley_jet_c_%d.png", i_snapshot), fig)
    end
end

make_animations = false
if make_animations
    create_panel_wise_visualization_animation(grid, cubed_sphere_bickley_jet_u_fields, framerate, "u"; k = Nz)
    create_panel_wise_visualization_animation(grid, cubed_sphere_bickley_jet_v_fields, framerate, "v"; k = Nz)
    create_panel_wise_visualization_animation(grid, cubed_sphere_bickley_jet_ζ_fields, framerate, "ζ"; k = Nz)
    create_panel_wise_visualization_animation(grid, cubed_sphere_bickley_jet_η_fields, framerate, "η"; k = Nz+1,
                                              ssh = true)
    create_panel_wise_visualization_animation(grid, cubed_sphere_bickley_jet_c_fields, framerate, "c"; k = Nz)

    prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]

    u_colorrange = specify_colorrange_timeseries(grid, u_fields)
    geo_heatlatlon_visualization_animation(grid, u_fields, "fc", prettytimes, "Zonal velocity"; k = Nz,
                                           cbar_label = "zonal velocity", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_bickley_jet_u_geo_heatlatlon_animation")

    v_colorrange = specify_colorrange_timeseries(grid, v_fields)
    geo_heatlatlon_visualization_animation(grid, v_fields, "cf", prettytimes, "Meridional velocity"; k = Nz,
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_bickley_jet_v_geo_heatlatlon_animation")

    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_fields)
    geo_heatlatlon_visualization_animation(grid, ζ_fields, "ff", prettytimes, "Relative vorticity"; k = Nz,
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_bickley_jet_ζ_geo_heatlatlon_animation")

    #=
    η_colorrange = specify_colorrange_timeseries(grid, η_fields; use_symmetric_colorrange = false, ssh = true)
    geo_heatlatlon_visualization_animation(grid, η_fields, "cc", prettytimes, "Surface elevation"; k = Nz+1,
                                           ssh = true, use_symmetric_colorrange = false,
                                           cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_bickley_jet_η_geo_heatlatlon_animation")
    =#

    c_colorrange = specify_colorrange_timeseries(grid, c_fields)
    geo_heatlatlon_visualization_animation(grid, c_fields, "cc", prettytimes, "Tracer distribution"; k = Nz,
                                           cbar_label = "tracer level", specify_plot_limits = true,
                                           plot_limits = c_colorrange, framerate = framerate,
                                           filename = "cubed_sphere_bickley_jet_c_geo_heatlatlon_animation")
end

using Oceananigans, Printf

using Oceananigans.Grids: λnode, φnode, znode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.Utils
using Oceananigans.Utils: Iterate
using KernelAbstractions: @kernel, @index

using JLD2

# Definition of the "Bickley jet": a sech(y)^2 jet with sinusoidal tracer
Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + (y + ℓ/10) / ℓ^2)
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    set_bickley_jet!(model; Ly = 4π, ϵ  = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

Set the `u` and `v`: Large-scale jet + vortical perturbations.
Set the tracer `c`: Sinusoid

Keyword args
============

* `Ly`: meridional domain extent
* `ϵ`: perturbation magnitude
* `ℓ₀`: Gaussian width for meridional extent of 4π
* `k₀`: sinusoidal wavenumber for domain extents of 4π in each direction
"""
function set_bickley_jet!(model; Ly = 4π, ϵ  = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

    ℓ = ℓ₀ / 4π * Ly
    k = k₀ * 4π / Ly

    dr(x) = deg2rad(x)

    ψᵢ(λ, φ, z) = ℓ * (Ψ(dr(φ)*8) + ϵ * ψ̃(dr(λ)*2, dr(φ)*8, ℓ, k))
    cᵢ(λ, φ, z) = C(dr(φ)*8, 180)

    ψ = Field{Face, Face, Center}(grid)

    # Note that set! fills only interior points; to compute u and v we need information in the halo regions.
    set!(ψ, ψᵢ)

    for region in [1, 3, 5]
        i = 1
        j = Ny+1
        for k in 1:Nz
            λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
            φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
            ψ[region][i, j, k] = ψᵢ(λ, φ, 0)
        end
    end

    for region in [2, 4, 6]
        i = Nx+1
        j = 1
        for k in 1:Nz
            λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
            φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
            ψ[region][i, j, k] = ψᵢ(λ, φ, 0)
        end
    end

    fill_halo_regions!(ψ)

    u = XFaceField(grid)
    v = YFaceField(grid)

    for region in 1:number_of_regions(grid)
        for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
            u[region][i, j, k] = - (ψ[region][i, j+1, k] - ψ[region][i, j, k]) / grid[region].Δyᶠᶜᵃ[i, j]
            v[region][i, j, k] =   (ψ[region][i+1, j, k] - ψ[region][i, j, k]) / grid[region].Δxᶜᶠᵃ[i, j]
        end
    end

    fill_halo_regions!((u, v))

    for region in 1:number_of_regions(grid)

        for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in 1:grid.Nz
            model.velocities.u[region][i, j, k] = u[region][i, j, k]
            model.velocities.v[region][i, j, k] = v[region][i, j, k]
        end

        for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
            λ = λnode(i, j, k, grid[region], Center(), Center(), Center())
            φ = φnode(i, j, k, grid[region], Center(), Center(), Center())
            z = znode(i, j, k, grid[region], Center(), Center(), Center())
            model.tracers.c[region][i, j, k] = cᵢ(λ, φ, z)
        end

    end

    fill_halo_regions!(model.tracers.c)

    return nothing
end

## Grid setup

H = 1000

Nx, Ny, Nz = 32, 32, 1
Nhalo = 4

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-H, 0), radius = 6370e3,
                                horizontal_direction_halo = Nhalo, partition = CubedSpherePartition(; R = 1))

Ly = 2π * grid.radius

Hx, Hy, Hz = halo_size(grid)

momentum_advection = VectorInvariant()
tracer_advection = WENO()
free_surface = ExplicitFreeSurface(gravitational_acceleration=10)

closure = ScalarDiffusivity(ν = 100 * 6370e3 * 0.0005)
# Using radius, r₁ = 1, number of points in each direction of a conformal cubed sphere panel, N = 32, and a time step,
# Δt₁ = 0.1 * min_spacing / c₁ = 0.1 * min_spacing / sqrt(g₁ * H₁), the numerical solution becomes unstable at
# ν = 0.0001, and experiences excessive diffusion at ν₁ = 0.001. So, we specify ν₁ = 0.0005, a value midway between
# 0.0001 and 0.001, to balance between retaining important gradient features and maintaining stability. For a much
# larger radius of r₂ = 6370e3, equivalent to Earth's radius, we apply scaling analysis and and obtain
# ν₂ = sqrt(c₂/c₁) r₂/r₁ = sqrt((g₂ H₂)/(g₁ H₁)) r₂/r₁ ν₁ = sqrt(10 * 1000) 6370e3/1 ν₁ = 100 * 6370e3 * 0.0005.

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection, free_surface, tracers = :c, 
                                    buoyancy=nothing, closure)

set_bickley_jet!(model; Ly = Ly, ϵ = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

# Specify cfl = c * Δt / min(Δx) = 0.2, so that Δt = 0.2 * min(Δx) / c

min_spacing = filter(!iszero, grid[1].Δxᶠᶠᵃ) |> minimum
c = sqrt(model.free_surface.gravitational_acceleration * H)
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
    Hx, Hy, Hz = halo_size(grid)

    fill_halo_regions!((sim.model.velocities.u, sim.model.velocities.v))

    offset = -1 .* halo_size(grid)

    @apply_regionally begin
        params = KernelParameters(total_size(ζ[1]), offset)
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, sim.model.velocities.u, sim.model.velocities.v)
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
for region in 1:number_of_regions(grid)
    for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in grid.Nz+1:grid.Nz+1
        ηᵢ[region][i, j, k] -= H
    end
end

cᵢ = deepcopy(simulation.model.tracers.c)

# Plot the initial velocity field.

include("cubed_sphere_visualization.jl")

fig = panel_wise_visualization_with_halos(grid, uᵢ)
save("u₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, uᵢ)
save("u₀.png", fig)

fig = panel_wise_visualization_with_halos(grid, vᵢ)
save("v₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, vᵢ)
save("v₀.png", fig)

# Plot the initial vorticity field after model definition.

fig = panel_wise_visualization_with_halos(grid, ζᵢ)
save("ζ₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, ζᵢ)
save("ζ₀.png", fig)

# Plot the initial surface elevation field after model definition.

fig = panel_wise_visualization_with_halos(grid, ηᵢ, grid.Nz+1, true, true)
save("η₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, ηᵢ, grid.Nz+1, true, true)
save("η₀.png", fig)

# Plot the initial tracer field.

fig = panel_wise_visualization_with_halos(grid, cᵢ)
save("c₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, cᵢ)
save("c₀.png", fig)

animation_time = 15 # seconds
framerate = 5
n_frames = animation_time * framerate
simulation_time_per_frame = stop_time/n_frames
# Specify animation_time and framerate in such a way that n_frames is a multiple of n_plots defined below.
save_fields_iteration_interval = floor(Int, simulation_time_per_frame/Δt)
# Redefine the simulation time per frame.
simulation_time_per_frame = save_fields_iteration_interval * Δt
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_ζ] = Callback(save_ζ, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_η] = Callback(save_η, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_c] = Callback(save_c, IterationInterval(save_fields_iteration_interval))

run!(simulation)

n_snapshots = length(η_fields)
for i_snapshot in 1:n_snapshots
    for region in 1:number_of_regions(grid)
        for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in grid.Nz+1:grid.Nz+1
            η_fields[i_snapshot][region][i, j, k] -= H
        end
    end
end

n_plots = 3

ζ_colorrange = zeros(2)
η_colorrange = zeros(2)
c_colorrange = zeros(2)

for i_plot in 1:n_plots
    frame_index = round(Int, i_plot * n_frames / n_plots)
    ζ_colorrange_at_frame_index = specify_colorrange(grid, ζ_fields[frame_index], true,  false)
    η_colorrange_at_frame_index = specify_colorrange(grid, η_fields[frame_index], false, true)
    c_colorrange_at_frame_index = specify_colorrange(grid, c_fields[frame_index], true,  false)
    if i_plot == 1
        ζ_colorrange[:] = collect(ζ_colorrange_at_frame_index)
        η_colorrange[:] = collect(η_colorrange_at_frame_index)
        c_colorrange[:] = collect(c_colorrange_at_frame_index)
    else
        ζ_colorrange[1] = min(ζ_colorrange[1], ζ_colorrange_at_frame_index[1])
        ζ_colorrange[2] = -ζ_colorrange[1]
        η_colorrange[1] = min(η_colorrange[1], η_colorrange_at_frame_index[1])
        η_colorrange[2] = max(η_colorrange[2], η_colorrange_at_frame_index[2])
        c_colorrange[1] = min(c_colorrange[1], c_colorrange_at_frame_index[1])
        c_colorrange[2] = -c_colorrange[1]
    end
end

for i_plot in 1:n_plots
    frame_index = round(Int, i_plot * n_frames / n_plots)
    simulation_time = simulation_time_per_frame * frame_index
    title = "Relative vorticity after $(prettytime(simulation_time*6371e3))"
    fig = geo_heatlatlon_visualization(grid, interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[frame_index],
                                                                                            "ff"), title;
                                       cbar_label = "Relative vorticity (s⁻¹)", specify_plot_limits = true,
                                       plot_limits = ζ_colorrange)
    save(@sprintf("ζ_%d.png", i_plot), fig)
    title = "Surface elevation after $(prettytime(simulation_time*6371e3))"
    fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; use_symmetric_colorrange = false, ssh = true,
                                       cbar_label = "Surface elevation (m)", specify_plot_limits = true,
                                       plot_limits = η_colorrange)
    save(@sprintf("η_%d.png", i_plot), fig)
    title = "Tracer distribution after $(prettytime(simulation_time*6371e3))"
    fig = geo_heatlatlon_visualization(grid, c_fields[frame_index], title;
                                       cbar_label = "Tracer level (tracer units m⁻³)", specify_plot_limits = true,
                                       plot_limits = c_colorrange)
    save(@sprintf("c_%d.png", i_plot), fig)
end

if print_output_to_jld2_file
    jldopen("cubed_sphere_bickley_jet_initial_condition.jld2", "w") do file
        for region in 1:6
            file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ
            file["u/" * string(region)] = u_fields[1][region][:, :, 1]
            file["v/" * string(region)] = v_fields[1][region][:, :, 1]
            file["ζ/" * string(region)] = ζ_fields[1][region][:, :, 1]
            file["η/" * string(region)] = η_fields[1][region][:, :, 1+1]
            file["c/" * string(region)] = c_fields[1][region][:, :, 1]
        end
    end
    jldopen("cubed_sphere_bickley_jet_output.jld2", "w") do file
        for region in 1:6
            file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ
            file["u/" * string(region)] = u_fields[end][region][:, :, 1]
            file["v/" * string(region)] = v_fields[end][region][:, :, 1]
            file["ζ/" * string(region)] = ζ_fields[end][region][:, :, 1]
            file["η/" * string(region)] = η_fields[end][region][:, :, 1+1]
            file["c/" * string(region)] = c_fields[end][region][:, :, 1]
        end
    end
end

#=
start_index = 1
use_symmetric_colorrange = true

create_panel_wise_visualization_animation(grid, u_fields, start_index, use_symmetric_colorrange, framerate, "u")
create_panel_wise_visualization_animation(grid, v_fields, start_index, use_symmetric_colorrange, framerate, "v")
create_panel_wise_visualization_animation(grid, ζ_fields, start_index, use_symmetric_colorrange, framerate, "ζ")
create_panel_wise_visualization_animation(grid, η_fields, start_index, use_symmetric_colorrange, framerate, "η",
                                          grid.Nz+1, true)
create_panel_wise_visualization_animation(grid, c_fields, start_index, use_symmetric_colorrange, framerate, "c")
=#

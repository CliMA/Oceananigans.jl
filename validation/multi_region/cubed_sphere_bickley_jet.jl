using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: xnode, ynode, znode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_paired_halo_regions!
using Oceananigans.Operators
using Oceananigans.Utils
using Oceananigans.Utils: Iterate
using KernelAbstractions: @kernel, @index
using DataDeps
using JLD2
using CairoMakie

include("cubed_sphere_visualization.jl")

# Definition of the "Bickley jet": a sech(y)^2 jet with sinusoidal tracer
Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    u, v: Large-scale jet + vortical perturbations
       c: Sinusoid
"""

function set_bickley_jet!(model;
                          Ly = π * 6371e3, # meridional domain extent
                          ϵ  = 0.1,        # perturbation magnitude
                          ℓ₀ = 0.5,        # Gaussian width for meridional extent of 4π
                          k₀ = 0.5)        # sinusoidal wavenumber for domain extents of 4π in each direction
    
    ℓ = ℓ₀/4π * Ly 
    k = k₀/4π * Ly
    
    # Initial conditions
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    cᵢ(x, y, z) = C(y, Ly)
    
    u = XFaceField(grid)
    v = YFaceField(grid)
    
    # Note that u, v are only horizontally-divergence-free as resolution -> ∞.
    for region in 1:number_of_regions(grid)
        for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
            x = xnode(i, j, k, grid[region], Face(), Face(), Center())
            y = ynode(i, j, k, grid[region], Face(), Face(), Center())
            z = znode(i, j, k, grid[region], Face(), Face(), Center())
            model.velocities.u[region][i, j, k] = uᵢ(x, y, z)
            x = xnode(i, j, k, grid[region], Center(), Face(), Center())
            y = ynode(i, j, k, grid[region], Center(), Face(), Center())
            z = znode(i, j, k, grid[region], Center(), Face(), Center())
            model.velocities.v[region][i, j, k] = vᵢ(x, y, z)
            x = xnode(i, j, k, grid[region], Center(), Center(), Center())
            y = ynode(i, j, k, grid[region], Center(), Center(), Center())
            z = znode(i, j, k, grid[region], Center(), Center(), Center())
            model.tracers.c[region][i, j, k] = cᵢ(x, y, z)
        end
    end
    
    fill_paired_halo_regions!((model.velocities.u, model.velocities.v))
    
    for _ in 1:3
        fill_halo_regions!(model.tracers.c)
    end

    return nothing
    
end

## Grid setup

R = 2
Ly = 2π * R
H = 1

Nx, Ny, Nz = 32, 32, 1
nHalo = 3

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-H, 0), radius = R, horizontal_direction_halo = nHalo,
                                partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

momentum_advection = VectorInvariant()
tracer_advection = WENO(order = 5)
free_surface = ImplicitFreeSurface(gravitational_acceleration=10.0)

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection, free_surface, tracers = :c, 
                                    buoyancy=nothing)

set_bickley_jet!(model; Ly = Ly, ϵ = 0.1, ℓ₀ = 0.5, k₀ = 0.5)

# Specify cfl = max|u| * Δt / min(Δx) = 0.2, so that Δt = 0.2 * min(Δx) / max|u|

min_spacing = filter(!iszero, grid[1].Δxᶠᶠᵃ) |> minimum
Δt = 0.2 * min_spacing / maximum(abs, model.velocities.u)
Ntime = 1000
stop_time = Ntime * Δt

simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f, max|c|: %.3f, 
                                wall time: %s\n", iteration(sim), prettytime(sim), prettytime(sim.Δt), 
                                maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η), 
                                maximum(abs, model.tracers.c), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

c_fields = Field[]
save_c(sim) = push!(c_fields, deepcopy(sim.model.tracers.c))

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

function save_vorticity(sim)

    Hx, Hy, Hz = halo_size(grid)

    fill_paired_halo_regions!((sim.model.velocities.u, sim.model.velocities.v))

    offset = -1 .* halo_size(grid)

    @apply_regionally begin
        params = KernelParameters(total_size(ζ[1]), offset)
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, sim.model.velocities.u, sim.model.velocities.v)
    end

    push!(ζ_fields, deepcopy(ζ))
    
end

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
save_fields_iteration_interval = floor(Int, stop_time/(Δt * n_frames))
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(save_fields_iteration_interval))

run!(simulation)

#=
start_index = 1
use_symmetric_colorrange = true

create_panel_wise_visualization_animation(grid, u_fields, start_index, use_symmetric_colorrange, framerate, "u")
create_panel_wise_visualization_animation(grid, v_fields, start_index, use_symmetric_colorrange, framerate, "v")
create_panel_wise_visualization_animation(grid, ζ_fields, start_index, use_symmetric_colorrange, framerate, "ζ")
create_panel_wise_visualization_animation(grid, v_fields, start_index, use_symmetric_colorrange, framerate, "η")
create_panel_wise_visualization_animation(grid, v_fields, start_index, use_symmetric_colorrange, framerate, "c")
=#
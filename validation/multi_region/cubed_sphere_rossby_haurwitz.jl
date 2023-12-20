using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_velocity_halos!
using Oceananigans.Operators
using Oceananigans.Utils: Iterate
#=
using Oceananigans.Diagnostics: accurate_cell_advection_timescale
=#
using JLD2
using CairoMakie

include("cubed_sphere_visualization.jl")

## Grid setup

R = 6371e3
H = 8000

#=
Nx, Ny, Nz = 5, 5, 1
=#
Nx, Ny, Nz = 32, 32, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-H, 0),
                                  radius = R,
                                  horizontal_direction_halo = 1,
                                  partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

# Fix the grid metric Δxᶠᶜᵃ[Nx+1,1-Hy:0] for odd panels.
for region in [1, 3, 5]
    region_east = region + 1
    grid[region].Δxᶠᶜᵃ[Nx+1,1-Hy:0] = reverse(grid[region_east].Δyᶜᶠᵃ[1:Hy,1])
end

# Fix the grid metric Δxᶠᶜᵃ[0,Ny+1:Ny+Hy] for even panels.
for region in [2, 4, 6]
    region_west = region - 1
    grid[region].Δxᶠᶜᵃ[0,Ny+1:Ny+Hy] = reverse(grid[region_west].Δyᶜᶠᵃ[Nx-Hy+1:Nx,Ny])
end

## Model setup

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface(; gravitational_acceleration = 100),
                                    coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving()),
                                    closure = nothing,
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
B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # Why not (n+1)^2 sin(θ)^2 + 1?
C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

ψ_function(θ, ϕ) = -R^2 * ω * sin(θ)^2 + R^2 * K * cos(θ)^n * sin(θ) * cos(n*ϕ)

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

for passes in 1:3
    fill_halo_regions!(ψ)
end

u = XFaceField(grid)
v = YFaceField(grid)

for region in 1:number_of_regions(grid)
    for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
        u[region][i, j, k] = - (ψ[region][i, j+1, k] - ψ[region][i, j, k]) / grid[region].Δyᶠᶜᵃ[i, j]
        v[region][i, j, k] =   (ψ[region][i+1, j, k] - ψ[region][i, j, k]) / grid[region].Δxᶜᶠᵃ[i, j]
        #=
        u[region][i, j, k] = i + Nx * (j - 1)
        v[region][i, j, k] = -100 - u[region][i, j, k]
        =#
        #=
        u[region][i, j, k] = 100*region + (i + Nx * (j - 1))
        v[region][i, j, k] = -u[region][i, j, k]
        =#
        #=
        u[region][i, j, k] = Nx*Ny*(region - 1) + (i + Nx * (j - 1))
        v[region][i, j, k] = -u[region][i, j, k]
        =#
    end
end

fill_velocity_halos!((; u, v, w = nothing))

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

plot_initial_condition_before_model_definition = true

if plot_initial_condition_before_model_definition
    # Plot the initial velocity field before model definition.

    fig = panel_wise_visualization_with_halos(grid, u)
    save("u₀₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u)
    save("u₀₀.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v)
    save("v₀₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v)
    save("v₀₀.png", fig)

    # Plot the initial vorticity field before model definition.

    fig = panel_wise_visualization_with_halos(grid, ζ)
    save("ζ₀₀_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ)
    save("ζ₀₀.png", fig)
end

jldopen("new_code.jld2", "w") do file
    for face in 1:6
        file["u/" * string(face)] = u.data[face]
        file["v/" * string(face)] = v.data[face]
    end
end

for region in 1:number_of_regions(grid)

    for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in 1:grid.Nz
        model.velocities.u[region][i,j,k] = u[region][i, j, k]
        model.velocities.v[region][i,j,k] = v[region][i, j, k]
    end
    
    for j in 1:grid.Ny, i in 1:grid.Nx, k in grid.Nz+1:grid.Nz+1
        λ = λnode(i, j, k, grid[region], Center(), Center(), Face())
        φ = φnode(i, j, k, grid[region], Center(), Center(), Face())
        model.free_surface.η[region][i, j, k] = η₀(λ, φ)
    end
    
end

for passes in 1:3
    fill_halo_regions!(model.free_surface.η)
end

## Simulation setup

# Compute amount of time needed for a 45° rotation.
angular_velocity = (n * (3+n) * ω - 2Ω) / ((1+n) * (2+n))
stop_time = deg2rad(360) / abs(angular_velocity)
@info "Stop time = $(prettytime(stop_time))"

Δt = 20

gravity_wave_speed = sqrt(g * H)
min_spacing = filter(!iszero, grid[1].Δyᶠᶠᵃ) |> minimum
wave_propagation_time_scale = min_spacing / gravity_wave_speed
gravity_wave_cfl = Δt / wave_propagation_time_scale
@info "Gravity wave CFL = $gravity_wave_cfl"

if !isnothing(model.closure)
    ν = model.closure.νh
    diffusive_cfl = ν * Δt / min_spacing^2
    @info "Diffusive CFL = $diffusive_cfl"
end

#=
cfl = CFL(Δt, accurate_cell_advection_timescale)
=#

cfl = 0.5

simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

ζ = Field{Face, Face, Center}(grid)

ζ_fields = Field[]
Δζ_fields = Field[]

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
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

# Plot the initial velocity field after model definition.

fig = panel_wise_visualization_with_halos(grid, uᵢ)
save("u₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, uᵢ)
save("u₀.png", fig)

fig = panel_wise_visualization_with_halos(grid, vᵢ)
save("v₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, vᵢ)
save("v₀.png", fig)

# Plot the initial surface elevation field after model definition.

fig = panel_wise_visualization_with_halos(grid, ηᵢ, grid.Nz+1, true, true)
save("η₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, ηᵢ, grid.Nz+1, true, true)
save("η₀.png", fig)

# Plot the initial vorticity field after model definition.

fig = panel_wise_visualization_with_halos(grid, ζᵢ)
save("ζ₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, ζᵢ)
save("ζ₀.png", fig)

function save_vorticity(sim)
    Hx, Hy, Hz = halo_size(grid)

    fill_velocity_halos!(sim.model.velocities)

    u, v, _ = sim.model.velocities
    
    offset = -1 .* halo_size(grid)
    
    @apply_regionally begin
        params = KernelParameters(total_size(ζ[1]), offset)
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(ζ_fields, deepcopy(ζ))
    
    Δζ_field = deepcopy(ζ)
    for region in 1:number_of_regions(grid)
        for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
            Δζ_field[region][i, j, k] -= ζᵢ[region][i, j, k]
        end
    end
    
    push!(Δζ_fields, Δζ_field)
end

save_fields_iteration_interval = 3
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(save_fields_iteration_interval))

#=
run!(simulation)
=#

#=
fig = panel_wise_visualization_with_halos(grid, Δζ_fields[end])
save("Δζ_with_halos.png", fig)

fig = panel_wise_visualization(grid, Δζ_fields[end])
save("Δζ.png", fig)

start_index = 1
use_symmetric_colorrange = true
animation_time = 10 # seconds
framerate = floor(Int, size(Δζ_fields)[1]/animation_time)

create_panel_wise_visualization_animation(grid, Δζ_fields, start_index, use_symmetric_colorrange, framerate, "Δζ")
=#
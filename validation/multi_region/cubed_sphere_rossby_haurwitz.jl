using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_velocity_halos!
using Oceananigans.Operators
using Oceananigans.Utils: Iterate
using JLD2
using CairoMakie

include("cubed_sphere_visualization.jl")

## Grid setup

R = 6371e3
H = 8000

Nx, Ny, Nz = 32, 32, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-H, 0),
                                  radius = R,
                                  horizontal_direction_halo = 1,
                                  partition = CubedSpherePartition(; R = 1))

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

A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2 )
B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # Why not (n+1)^2 sin(θ)^2 + 1?
C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

ψ_function(θ, ϕ) = -R^2 * ω * sin(θ)^2 + R^2 * K * cos(θ)^n * sin(θ) * cos(n*ϕ)

u_function(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ)
v_function(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ)

h_function(θ, ϕ) = H + R^2/g * (A(θ) + B(θ) * cos(n * ϕ) + C(θ) * cos(2n * ϕ))

# Initial conditions
# Previously: θ ∈ [-π/2, π/2] is latitude and ϕ ∈ [0, 2π) is longitude
# Oceananigans: ϕ ∈ [-90, 90] and λ ∈ [-180, 180]

rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ

# Arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
u₀(λ, ϕ, z) = u_function(rescale²(ϕ), rescale¹(λ))
v₀(λ, ϕ, z) = v_function(rescale²(ϕ), rescale¹(λ))
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
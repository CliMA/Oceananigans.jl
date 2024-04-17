#=
Download:
(a) the file old_code_metrics.jld2 from
    https://www.dropbox.com/scl/fo/qu7nfr94wqc6ym6izpfqw/h?rlkey=zd4o5134u64ibyxggy64tiygt&dl=0; and
(b) the directory grid_cs32+ol4 from 
    https://www.dropbox.com/scl/fo/c0pex0u8yvao6ehd3rqtp/h?rlkey=uq8bojrrsa7c4pb4n9ou8wvcs&dl=0;
and place them in the path validation/multi_region/. Then run this script from the same path as:
include("cubed_sphere_rossby_haurwitz.jl")
=#

using Oceananigans, Printf

using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.Utils: Iterate
using DataDeps
using JLD2
using CairoMakie

include("cubed_sphere_visualization.jl")

## Grid setup

R = 6371e3
H = 8000

load_cs32_grid = false

if load_cs32_grid
    dd32 = DataDep("cubed_sphere_32_grid",
                   "Conformal cubed sphere grid with 32×32 grid points on each face",
                   "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
                   "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538")
    DataDeps.register(dd32)
    grid_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
    grid = ConformalCubedSphereGrid(grid_filepath; 
                                    Nz = 1,
                                    z = (-H, 0),
                                    panel_halo = (1, 1, 1),
                                    radius = R)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
else
    old_code_metrics_JMC = true
    if old_code_metrics_JMC
        Nx, Ny, Nz = 32, 32, 1
        print_jmc_cs32_grid = false
        if print_jmc_cs32_grid
            Nhalo = 4
        else
            Nhalo = 1 # For the purpose of comparing metrics, you may choose any integer from 1 to 4.
        end
    else
        Nx, Ny, Nz = 32, 32, 1
        Nhalo = 1
    end
    grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                      z = (-H, 0),
                                      radius = R,
                                      horizontal_direction_halo = Nhalo,
                                      partition = CubedSpherePartition(; R = 1))
end

Hx, Hy, Hz = halo_size(grid)

grid_λᶜᶜᵃ  = Field{Center, Center, Center}(grid)
grid_λᶠᶠᵃ  = Field{Face,   Face,   Center}(grid)
grid_φᶜᶜᵃ  = Field{Center, Center, Center}(grid)
grid_φᶠᶠᵃ  = Field{Face,   Face,   Center}(grid)
grid_Δxᶠᶜᵃ = Field{Face,   Center, Center}(grid)
grid_Δyᶜᶠᵃ = Field{Center, Face,   Center}(grid)
grid_Azᶜᶜᵃ = Field{Center, Center, Center}(grid)
grid_Azᶠᶜᵃ = Field{Face,   Center, Center}(grid)
grid_Azᶜᶠᵃ = Field{Center, Face,   Center}(grid)
grid_Azᶠᶠᵃ = Field{Face,   Face,   Center}(grid)

for region in 1:6
    for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy, k in 1:Nz
        grid_λᶜᶜᵃ[region][i, j, k]  = grid[region].λᶜᶜᵃ[i, j]
        grid_λᶠᶠᵃ[region][i, j, k]  = grid[region].λᶠᶠᵃ[i, j]
        grid_φᶜᶜᵃ[region][i, j, k]  = grid[region].φᶜᶜᵃ[i, j]
        grid_φᶠᶠᵃ[region][i, j, k]  = grid[region].φᶠᶠᵃ[i, j]
        grid_Δxᶠᶜᵃ[region][i, j, k] = grid[region].Δxᶠᶜᵃ[i, j]
        grid_Δyᶜᶠᵃ[region][i, j, k] = grid[region].Δyᶜᶠᵃ[i, j]
        grid_Azᶜᶜᵃ[region][i, j, k] = Azᶜᶜᶜ(i, j, k, grid[region])
        grid_Azᶠᶜᵃ[region][i, j, k] = Azᶠᶜᶜ(i, j, k, grid[region])
        grid_Azᶜᶠᵃ[region][i, j, k] = Azᶜᶠᶜ(i, j, k, grid[region])
        grid_Azᶠᶠᵃ[region][i, j, k] = Azᶠᶠᶜ(i, j, k, grid[region])
    end
end

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

# Plot the grid metrics.

fig = panel_wise_visualization_with_halos(grid, grid_λᶜᶜᵃ)
save("grid_λᶜᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_λᶜᶜᵃ)
save("grid_λᶜᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_λᶠᶠᵃ)
save("grid_λᶠᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_λᶠᶠᵃ)
save("grid_λᶠᶠᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_φᶜᶜᵃ)
save("grid_φᶜᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_φᶜᶜᵃ)
save("grid_φᶜᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_φᶠᶠᵃ)
save("grid_φᶠᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_φᶠᶠᵃ)
save("grid_φᶠᶠᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Δxᶠᶜᵃ)
save("grid_Δxᶠᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Δxᶠᶜᵃ)
save("grid_Δxᶠᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Δyᶜᶠᵃ)
save("grid_Δyᶜᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Δyᶜᶠᵃ)
save("grid_Δyᶜᶠᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Azᶜᶜᵃ)
save("grid_Azᶜᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Azᶜᶜᵃ)
save("grid_Azᶜᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Azᶠᶜᵃ)
save("grid_Azᶠᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Azᶠᶜᵃ)
save("grid_Azᶠᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Azᶜᶠᵃ)
save("grid_Azᶜᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Azᶜᶠᵃ)
save("grid_Azᶜᶠᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Azᶠᶠᵃ)
save("grid_Azᶠᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Azᶠᶠᵃ)
save("grid_Azᶠᶠᵃ.png", fig)

plot_initial_condition_before_model_definition = false

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

jldopen("new_code_metrics.jld2", "w") do file
    for region in 1:6
        file["λᶜᶜᵃ/" * string(region)]  =  grid[region].λᶜᶜᵃ
        file["λᶠᶠᵃ/" * string(region)]  =  grid[region].λᶠᶠᵃ
        file["φᶜᶜᵃ/" * string(region)]  =  grid[region].φᶜᶜᵃ
        file["φᶠᶠᵃ/" * string(region)]  =  grid[region].φᶠᶠᵃ
        file["Δxᶠᶜᵃ/" * string(region)] = grid[region].Δxᶠᶜᵃ
        file["Δxᶜᶠᵃ/" * string(region)] = grid[region].Δxᶜᶠᵃ
        file["Δyᶠᶜᵃ/" * string(region)] = grid[region].Δyᶠᶜᵃ
        file["Δyᶜᶠᵃ/" * string(region)] = grid[region].Δyᶜᶠᵃ
        file["Azᶜᶜᵃ/" * string(region)] = grid[region].Azᶜᶜᵃ
        file["Azᶠᶜᵃ/" * string(region)] = grid[region].Azᶠᶜᵃ
        file["Azᶜᶠᵃ/" * string(region)] = grid[region].Azᶜᶠᵃ
        file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ       
    end
end

compare_old_and_new_code_metrics = false

if compare_old_and_new_code_metrics

    old_λᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_λᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_φᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_φᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δxᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δxᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δyᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δyᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶜᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶠᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    
    new_λᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_λᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_φᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_φᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Δxᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Δxᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Δyᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Δyᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Azᶜᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Azᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Azᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    new_Azᶠᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)

    if old_code_metrics_JMC
        for region in 1:6
            old_λᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/xC.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_λᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/xG.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_φᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/yC.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_φᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/yG.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Δxᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dXc.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Δxᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dXg.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Δyᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dYg.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Δyᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dYc.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Azᶜᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAc.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Azᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAw.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Azᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAs.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
            old_Azᶠᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAz.00$(region).001.data", 32, 4)[1+4-Nhalo:end-4+Nhalo,1+4-Nhalo:end-4+Nhalo]
        end
        if print_jmc_cs32_grid
            jldopen("jmc_cubed_sphere_32_grid_with_4_halos.jld2", "w") do file
                for region in 1:6
                    file["face" * string(region) * "/λᶜᶜᵃ" ] =  old_λᶜᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/λᶠᶠᵃ" ] =  old_λᶠᶠᵃ_parent[:, :, region]
                    file["face" * string(region) * "/φᶜᶜᵃ" ] =  old_φᶜᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/φᶠᶠᵃ" ] =  old_φᶠᶠᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Δxᶠᶜᵃ"] = old_Δxᶠᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Δxᶜᶠᵃ"] = old_Δxᶜᶠᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Δyᶠᶜᵃ"] = old_Δyᶠᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Δyᶜᶠᵃ"] = old_Δyᶜᶠᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Azᶜᶜᵃ"] = old_Azᶜᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Azᶠᶜᵃ"] = old_Azᶠᶜᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Azᶜᶠᵃ"] = old_Azᶜᶠᵃ_parent[:, :, region]
                    file["face" * string(region) * "/Azᶠᶠᵃ"] = old_Azᶠᶠᵃ_parent[:, :, region]
                    # Fill the following metrics with their Oceananigans counterparts for now.
                    file["face" * string(region) * "/Δxᶜᶜᵃ"] = parent(grid[region].Δxᶜᶜᵃ)
                    file["face" * string(region) * "/Δyᶜᶜᵃ"] = parent(grid[region].Δyᶜᶜᵃ)
                    file["face" * string(region) * "/Δxᶠᶠᵃ"] = parent(grid[region].Δxᶠᶠᵃ)
                    file["face" * string(region) * "/Δyᶠᶠᵃ"] = parent(grid[region].Δyᶠᶠᵃ)
                end
            end
        end
    else    
        old_file = jldopen("old_code_metrics.jld2")
        for region in 1:6
            old_Δxᶠᶜᵃ_parent[:, :, region] = parent(old_file["Δxᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :])
            old_Δxᶜᶠᵃ_parent[:, :, region] = parent(old_file["Δxᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy])
            old_Δyᶠᶜᵃ_parent[:, :, region] = parent(old_file["Δyᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :])
            old_Δyᶜᶠᵃ_parent[:, :, region] = parent(old_file["Δyᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy])
            old_Azᶜᶜᵃ_parent[:, :, region] = parent(old_file["Azᶜᶜᵃ/" * string(region)][:, :])
            old_Azᶠᶜᵃ_parent[:, :, region] = parent(old_file["Azᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :])
            old_Azᶜᶠᵃ_parent[:, :, region] = parent(old_file["Azᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy])
            old_Azᶠᶠᵃ_parent[:, :, region] = parent(old_file["Azᶠᶠᵃ/" * string(region)][1-Hx:Nx+Hx, 1-Hy:Ny+Hy])
        end
    end

    overwrite_grid_metrics_from_old_code = false
    if overwrite_grid_metrics_from_old_code
        if old_code_metrics_JMC
            for region in 1:6
                grid[region].λᶜᶜᵃ[:,:]  =  old_λᶜᶜᵃ_parent[:, :, region]
                grid[region].λᶠᶠᵃ[:,:]  =  old_λᶠᶠᵃ_parent[:, :, region]
                grid[region].φᶜᶜᵃ[:,:]  =  old_φᶜᶜᵃ_parent[:, :, region]
                grid[region].φᶠᶠᵃ[:,:]  =  old_φᶠᶠᵃ_parent[:, :, region]
                grid[region].Δxᶠᶜᵃ[:,:] = old_Δxᶠᶜᵃ_parent[:, :, region]
                grid[region].Δxᶜᶠᵃ[:,:] = old_Δxᶜᶠᵃ_parent[:, :, region]
                grid[region].Δyᶠᶜᵃ[:,:] = old_Δyᶠᶜᵃ_parent[:, :, region]
                grid[region].Δyᶜᶠᵃ[:,:] = old_Δyᶜᶠᵃ_parent[:, :, region]
                grid[region].Azᶜᶜᵃ[:,:] = old_Azᶜᶜᵃ_parent[:, :, region]
                grid[region].Azᶠᶜᵃ[:,:] = old_Azᶠᶜᵃ_parent[:, :, region]
                grid[region].Azᶜᶠᵃ[:,:] = old_Azᶜᶠᵃ_parent[:, :, region]
                grid[region].Azᶠᶠᵃ[:,:] = old_Azᶠᶠᵃ_parent[:, :, region]
            end
        else
            for region in 1:6
                grid[region].Δxᶠᶜᵃ[:,:] = old_file["Δxᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :]
                grid[region].Δxᶜᶠᵃ[:,:] = old_file["Δxᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy]
                grid[region].Δyᶠᶜᵃ[:,:] = old_file["Δyᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :]
                grid[region].Δyᶜᶠᵃ[:,:] = old_file["Δyᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy]
                grid[region].Azᶜᶜᵃ[:,:] = old_file["Azᶜᶜᵃ/" * string(region)][:, :]
                grid[region].Azᶠᶜᵃ[:,:] = old_file["Azᶠᶜᵃ/" * string(region)][1-Hx:Nx+Hx, :]
                grid[region].Azᶜᶠᵃ[:,:] = old_file["Azᶜᶠᵃ/" * string(region)][:, 1-Hy:Ny+Hy]
                grid[region].Azᶠᶠᵃ[:,:] = old_file["Azᶠᶠᵃ/" * string(region)][1-Hx:Nx+Hx, 1-Hy:Ny+Hy]
            end
        end
    end
    
    if !old_code_metrics_JMC
        close(old_file)
    end
    
    new_file = jldopen("new_code_metrics.jld2")
    for region in 1:6
        new_λᶜᶜᵃ_parent[:, :, region]  =  parent(new_file["λᶜᶜᵃ/" * string(region)][:, :, 1])
        new_λᶠᶠᵃ_parent[:, :, region]  =  parent(new_file["λᶠᶠᵃ/" * string(region)][:, :, 1])
        new_φᶜᶜᵃ_parent[:, :, region]  =  parent(new_file["φᶜᶜᵃ/" * string(region)][:, :, 1])
        new_φᶠᶠᵃ_parent[:, :, region]  =  parent(new_file["φᶠᶠᵃ/" * string(region)][:, :, 1])
        new_Δxᶠᶜᵃ_parent[:, :, region] = parent(new_file["Δxᶠᶜᵃ/" * string(region)][:, :, 1])
        new_Δxᶜᶠᵃ_parent[:, :, region] = parent(new_file["Δxᶜᶠᵃ/" * string(region)][:, :, 1])
        new_Δyᶠᶜᵃ_parent[:, :, region] = parent(new_file["Δyᶠᶜᵃ/" * string(region)][:, :, 1])
        new_Δyᶜᶠᵃ_parent[:, :, region] = parent(new_file["Δyᶜᶠᵃ/" * string(region)][:, :, 1])
        new_Azᶜᶜᵃ_parent[:, :, region] = parent(new_file["Azᶜᶜᵃ/" * string(region)][:, :, 1])
        new_Azᶠᶜᵃ_parent[:, :, region] = parent(new_file["Azᶠᶜᵃ/" * string(region)][:, :, 1])
        new_Azᶜᶠᵃ_parent[:, :, region] = parent(new_file["Azᶜᶠᵃ/" * string(region)][:, :, 1])
        new_Azᶠᶠᵃ_parent[:, :, region] = parent(new_file["Azᶠᶠᵃ/" * string(region)][:, :, 1])
    end
    close(new_file)
    
    λᶜᶜᵃ_difference  =  new_λᶜᶜᵃ_parent - old_λᶜᶜᵃ_parent
    λᶠᶠᵃ_difference  =  new_λᶠᶠᵃ_parent - old_λᶠᶠᵃ_parent
    φᶜᶜᵃ_difference  =  new_φᶜᶜᵃ_parent - old_φᶜᶜᵃ_parent
    φᶠᶠᵃ_difference  =  new_φᶠᶠᵃ_parent - old_φᶠᶠᵃ_parent
    Δxᶠᶜᵃ_difference = new_Δxᶠᶜᵃ_parent - old_Δxᶠᶜᵃ_parent
    Δxᶜᶠᵃ_difference = new_Δxᶜᶠᵃ_parent - old_Δxᶜᶠᵃ_parent
    Δyᶠᶜᵃ_difference = new_Δyᶠᶜᵃ_parent - old_Δyᶠᶜᵃ_parent
    Δyᶜᶠᵃ_difference = new_Δyᶜᶠᵃ_parent - old_Δyᶜᶠᵃ_parent
    Azᶜᶜᵃ_difference = new_Azᶜᶜᵃ_parent - old_Azᶜᶜᵃ_parent
    Azᶠᶜᵃ_difference = new_Azᶠᶜᵃ_parent - old_Azᶠᶜᵃ_parent
    Azᶜᶠᵃ_difference = new_Azᶜᶠᵃ_parent - old_Azᶜᶠᵃ_parent
    Azᶠᶠᵃ_difference = new_Azᶠᶠᵃ_parent - old_Azᶠᶠᵃ_parent
    
    λᶜᶜᵃ_relative_difference  =  λᶜᶜᵃ_difference ./ old_λᶜᶜᵃ_parent
    λᶠᶠᵃ_relative_difference  =  λᶠᶠᵃ_difference ./ old_λᶠᶠᵃ_parent
    φᶜᶜᵃ_relative_difference  =  φᶜᶜᵃ_difference ./ old_φᶜᶜᵃ_parent
    φᶠᶠᵃ_relative_difference  =  φᶠᶠᵃ_difference ./ old_φᶠᶠᵃ_parent
    Δxᶠᶜᵃ_relative_difference = Δxᶠᶜᵃ_difference ./ old_Δxᶠᶜᵃ_parent
    Δxᶜᶠᵃ_relative_difference = Δxᶜᶠᵃ_difference ./ old_Δxᶜᶠᵃ_parent
    Δyᶠᶜᵃ_relative_difference = Δyᶠᶜᵃ_difference ./ old_Δyᶠᶜᵃ_parent
    Δyᶜᶠᵃ_relative_difference = Δyᶜᶠᵃ_difference ./ old_Δyᶜᶠᵃ_parent
    Azᶜᶜᵃ_relative_difference = Azᶜᶜᵃ_difference ./ old_Azᶜᶜᵃ_parent
    Azᶠᶜᵃ_relative_difference = Azᶠᶜᵃ_difference ./ old_Azᶠᶜᵃ_parent
    Azᶜᶠᵃ_relative_difference = Azᶜᶠᵃ_difference ./ old_Azᶜᶠᵃ_parent
    Azᶠᶠᵃ_relative_difference = Azᶠᶠᵃ_difference ./ old_Azᶠᶠᵃ_parent

    λᶜᶜᵃ_relative_difference[ old_λᶜᶜᵃ_parent  .== 0] .= 0
    λᶠᶠᵃ_relative_difference[ old_λᶠᶠᵃ_parent  .== 0] .= 0
    φᶜᶜᵃ_relative_difference[ old_φᶜᶜᵃ_parent  .== 0] .= 0
    φᶠᶠᵃ_relative_difference[ old_φᶠᶠᵃ_parent  .== 0] .= 0
    Δxᶠᶜᵃ_relative_difference[old_Δxᶠᶜᵃ_parent .== 0] .= 0
    Δxᶜᶠᵃ_relative_difference[old_Δxᶜᶠᵃ_parent .== 0] .= 0
    Δyᶠᶜᵃ_relative_difference[old_Δyᶠᶜᵃ_parent .== 0] .= 0
    Δyᶜᶠᵃ_relative_difference[old_Δyᶜᶠᵃ_parent .== 0] .= 0
    Azᶜᶜᵃ_relative_difference[old_Azᶜᶜᵃ_parent .== 0] .= 0
    Azᶠᶜᵃ_relative_difference[old_Azᶠᶜᵃ_parent .== 0] .= 0
    Azᶜᶠᵃ_relative_difference[old_Azᶜᶠᵃ_parent .== 0] .= 0
    Azᶠᶠᵃ_relative_difference[old_Azᶠᶠᵃ_parent .== 0] .= 0
    
end

jldopen("new_code.jld2", "w") do file
    for region in 1:6
        file["u/" * string(region)] = u.data[region]
        file["v/" * string(region)] = v.data[region]
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
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u),
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
Δζ_fields = Field[]

function save_ζ(sim)
    Hx, Hy, Hz = halo_size(grid)

    fill_halo_regions!((sim.model.velocities.u, sim.model.velocities.v))

    offset = -1 .* halo_size(grid)

    @apply_regionally begin
        params = KernelParameters(total_size(ζ[1]), offset)
        launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, sim.model.velocities.u, sim.model.velocities.v)
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

η_fields = Field[]
save_η(sim) = push!(η_fields, deepcopy(sim.model.free_surface.η))

uᵢ = deepcopy(simulation.model.velocities.u)
vᵢ = deepcopy(simulation.model.velocities.v)
ζᵢ = deepcopy(ζ) 

ηᵢ = deepcopy(simulation.model.free_surface.η)
for region in 1:number_of_regions(grid)
    for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in grid.Nz+1:grid.Nz+1
        ηᵢ[region][i, j, k] -= H
    end
end

if compare_old_and_new_code_metrics

    # Plot the relative difference of the grid metrics with halos.

    fig = panel_wise_visualization_of_grid_metrics_with_halos(λᶜᶜᵃ_relative_difference)
    save("λᶜᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(λᶠᶠᵃ_relative_difference)
    save("λᶠᶠᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(φᶜᶜᵃ_relative_difference)
    save("φᶜᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(φᶠᶠᵃ_relative_difference)
    save("φᶠᶠᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Δxᶠᶜᵃ_relative_difference)
    save("Δxᶠᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Δxᶜᶠᵃ_relative_difference)
    save("Δxᶜᶠᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Δyᶠᶜᵃ_relative_difference)
    save("Δyᶠᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Δyᶜᶠᵃ_relative_difference)
    save("Δyᶜᶠᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Azᶜᶜᵃ_relative_difference)
    save("Azᶜᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Azᶠᶜᵃ_relative_difference)
    save("Azᶠᶜᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Azᶜᶠᵃ_relative_difference)
    save("Azᶜᶠᵃ_relative_difference_with_halos.png", fig)

    fig = panel_wise_visualization_of_grid_metrics_with_halos(Azᶠᶠᵃ_relative_difference)
    save("Azᶠᶠᵃ_relative_difference_with_halos.png", fig)

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

for i_plot in 1:n_plots
    frame_index = round(Int, i_plot * n_frames / n_plots)
    ζ_colorrange_at_frame_index = specify_colorrange(grid, ζ_fields[frame_index], true, false)
    η_colorrange_at_frame_index = specify_colorrange(grid, η_fields[frame_index], true, true)
    if i_plot == 1
        ζ_colorrange[:] = collect(ζ_colorrange_at_frame_index)
        η_colorrange[:] = collect(η_colorrange_at_frame_index)
    else
        ζ_colorrange[1] = min(ζ_colorrange[1], ζ_colorrange_at_frame_index[1])
        ζ_colorrange[2] = -ζ_colorrange[1]
        η_colorrange[1] = min(η_colorrange[1], η_colorrange_at_frame_index[1])
        η_colorrange[2] = -η_colorrange[1]
    end
end

for i_plot in 1:n_plots
    frame_index = round(Int, i_plot * n_frames / n_plots)
    simulation_time = simulation_time_per_frame * frame_index
    title = "Relative vorticity after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid,
                                       interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[frame_index],
                                                                                      "ff"), title;
                                       cbar_label = "Relative vorticity (s⁻¹)", specify_plot_limits = true,
                                       plot_limits = ζ_colorrange)
    save(@sprintf("ζ_%d.png", i_plot), fig)
    title = "Surface elevation after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; ssh = true,
                                       cbar_label = "Surface elevation (m)", specify_plot_limits = true,
                                       plot_limits = η_colorrange)
    save(@sprintf("η_%d.png", i_plot), fig)
end

if print_output_to_jld2_file
    jldopen("cubed_sphere_rossby_haurwitz_initial_condition.jld2", "w") do file
        for region in 1:6
            file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ
            file["u/" * string(region)] = u_fields[1][region][:, :, 1]
            file["v/" * string(region)] = v_fields[1][region][:, :, 1]
            file["ζ/" * string(region)] = ζ_fields[1][region][:, :, 1]
            file["η/" * string(region)] = η_fields[1][region][:, :, 1+1]
        end
    end
    jldopen("cubed_sphere_rossby_haurwitz_output.jld2", "w") do file
        for region in 1:6
            file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ
            file["u/" * string(region)] = u_fields[end][region][:, :, 1]
            file["v/" * string(region)] = v_fields[end][region][:, :, 1]
            file["ζ/" * string(region)] = ζ_fields[end][region][:, :, 1]
            file["η/" * string(region)] = η_fields[end][region][:, :, 1+1]
        end
    end
end

#=
fig = panel_wise_visualization_with_halos(grid, u_fields[end])
save("u_with_halos.png", fig)

fig = panel_wise_visualization(grid, u_fields[end])
save("u.png", fig)

fig = panel_wise_visualization_with_halos(grid, v_fields[end])
save("v_with_halos.png", fig)

fig = panel_wise_visualization(grid, v_fields[end])
save("v.png", fig)

fig = panel_wise_visualization_with_halos(grid, ζ_fields[end])
save("ζ_with_halos.png", fig)

fig = panel_wise_visualization(grid, ζ_fields[end])
save("ζ.png", fig)

fig = panel_wise_visualization_with_halos(grid, Δζ_fields[end])
save("Δζ_with_halos.png", fig)

fig = panel_wise_visualization(grid, Δζ_fields[end])
save("Δζ.png", fig)

start_index = 1
use_symmetric_colorrange = true

create_panel_wise_visualization_animation(grid, u_fields, start_index, use_symmetric_colorrange, framerate, "u")
create_panel_wise_visualization_animation(grid, v_fields, start_index, use_symmetric_colorrange, framerate, "v")
create_panel_wise_visualization_animation(grid, ζ_fields, start_index, use_symmetric_colorrange, framerate, "ζ")
create_panel_wise_visualization_animation(grid, Δζ_fields, start_index, use_symmetric_colorrange, framerate, "Δζ")
=#

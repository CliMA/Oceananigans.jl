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
        nHalo = 1 # For the purpose of comparing metrics, you may choose any integer from 1 to 4.
    else
        #=
        Nx, Ny, Nz = 5, 5, 1
        =#
        Nx, Ny, Nz = 32, 32, 1
        nHalo = 1
    end
    grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                      z = (-H, 0),
                                      radius = R,
                                      horizontal_direction_halo = nHalo,
                                      partition = CubedSpherePartition(; R = 1))
end

Hx, Hy, Hz = halo_size(grid)

grid_Δxᶠᶜᵃ = Field{Face, Center, Center}(grid)
grid_Δyᶜᶠᵃ = Field{Center, Face, Center}(grid)
grid_Azᶠᶠᵃ = Field{Face, Face, Center}(grid)

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

for region in 1:6
    for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy, k in 1:Nz
        grid_Δxᶠᶜᵃ[region][i, j, k] = grid[region].Δxᶠᶜᵃ[i, j]
        grid_Δyᶜᶠᵃ[region][i, j, k] = grid[region].Δyᶜᶠᵃ[i, j]
        grid_Azᶠᶠᵃ[region][i, j, k] = Azᶠᶠᶜ(i, j, k, grid[region])
    end
end

## Model setup

horizontal_closure = HorizontalScalarDiffusivity(ν = 1e+4) 
#=
Switch between horizontal_closure = HorizontalScalarDiffusivity(ν = 1e+4) and horizontal_closure = nothing. Here, ν is 
the horizontal viscosity for the momentum equations and κ is the horizontal diffusivity for the continuity equation. 
Both are in [m² s⁻¹].
=#

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

# Plot the grid metrics.

fig = panel_wise_visualization_with_halos(grid, grid_Δxᶠᶜᵃ)
save("grid_Δxᶠᶜᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Δxᶠᶜᵃ)
save("grid_Δxᶠᶜᵃ.png", fig)

fig = panel_wise_visualization_with_halos(grid, grid_Δyᶜᶠᵃ)
save("grid_Δyᶜᶠᵃ_with_halos.png", fig)

fig = panel_wise_visualization(grid, grid_Δyᶜᶠᵃ)
save("grid_Δyᶜᶠᵃ.png", fig)

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

compare_old_and_new_code_metrics = true

if compare_old_and_new_code_metrics

    old_xᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_xᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_yᶜᶜᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_yᶠᶠᵃ_parent  = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δxᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δxᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δyᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Δyᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶜᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶠᶜᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶜᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    old_Azᶠᶠᵃ_parent = zeros(Nx+2Hx, Ny+2Hy, 6)
    
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
            old_xᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/XC.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_xᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/XG.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_yᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/YC.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_yᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/YG.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Δxᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dXc.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Δxᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dXg.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Δyᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dYg.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Δyᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/dYc.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Azᶜᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAc.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Azᶠᶜᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAw.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Azᶜᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAs.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_Azᶠᶠᵃ_parent[:, :, region] = read_big_endian_coordinates("grid_cs32+ol4/rAz.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
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

    overwrite_grid_metrics_from_old_code = true
    if overwrite_grid_metrics_from_old_code
        if old_code_metrics_JMC
            for region in 1:6
                grid[region].λᶜᶜᵃ[:,:]  =  old_xᶜᶜᵃ_parent[:, :, region]
                grid[region].λᶠᶠᵃ[:,:]  =  old_xᶠᶠᵃ_parent[:, :, region]
                grid[region].φᶜᶜᵃ[:,:]  =  old_yᶜᶜᵃ_parent[:, :, region]
                grid[region].φᶠᶠᵃ[:,:]  =  old_yᶠᶠᵃ_parent[:, :, region]
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
    
    Δxᶠᶜᵃ_difference = new_Δxᶠᶜᵃ_parent - old_Δxᶠᶜᵃ_parent
    Δxᶜᶠᵃ_difference = new_Δxᶜᶠᵃ_parent - old_Δxᶜᶠᵃ_parent
    Δyᶠᶜᵃ_difference = new_Δyᶠᶜᵃ_parent - old_Δyᶠᶜᵃ_parent
    Δyᶜᶠᵃ_difference = new_Δyᶜᶠᵃ_parent - old_Δyᶜᶠᵃ_parent
    Azᶜᶜᵃ_difference = new_Azᶜᶜᵃ_parent - old_Azᶜᶜᵃ_parent
    Azᶠᶜᵃ_difference = new_Azᶠᶜᵃ_parent - old_Azᶠᶜᵃ_parent
    Azᶜᶠᵃ_difference = new_Azᶜᶠᵃ_parent - old_Azᶜᶠᵃ_parent
    Azᶠᶠᵃ_difference = new_Azᶠᶠᵃ_parent - old_Azᶠᶠᵃ_parent
    
    Δxᶠᶜᵃ_relative_difference = Δxᶠᶜᵃ_difference ./ old_Δxᶠᶜᵃ_parent
    Δxᶜᶠᵃ_relative_difference = Δxᶜᶠᵃ_difference ./ old_Δxᶜᶠᵃ_parent
    Δyᶠᶜᵃ_relative_difference = Δyᶠᶜᵃ_difference ./ old_Δyᶠᶜᵃ_parent
    Δyᶜᶠᵃ_relative_difference = Δyᶜᶠᵃ_difference ./ old_Δyᶜᶠᵃ_parent
    Azᶜᶜᵃ_relative_difference = Azᶜᶜᵃ_difference ./ old_Azᶜᶜᵃ_parent
    Azᶠᶜᵃ_relative_difference = Azᶠᶜᵃ_difference ./ old_Azᶠᶜᵃ_parent
    Azᶜᶠᵃ_relative_difference = Azᶜᶠᵃ_difference ./ old_Azᶜᶠᵃ_parent
    Azᶠᶠᵃ_relative_difference = Azᶠᶠᵃ_difference ./ old_Azᶠᶠᵃ_parent
    
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

for passes in 1:3
    fill_halo_regions!(model.free_surface.η)
end

## Simulation setup

# Compute amount of time needed for a 45° rotation.
angular_velocity = (n * (3+n) * ω - 2Ω) / ((1+n) * (2+n))
stop_time = deg2rad(360) / abs(angular_velocity)
@info "Stop time = $(prettytime(stop_time))"

Δt = 20 # The numerical solution blows up, not just with Δt = 20, but also with Δt = 5, a factor of 4 less.

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

#=
cfl = CFL(Δt, accurate_cell_advection_timescale)
=#

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

animation_time = 15 # seconds
framerate = 5
n_frames = animation_time * framerate
save_fields_iteration_interval = floor(Int, stop_time/(Δt * n_frames))
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(save_fields_iteration_interval))

run!(simulation)

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
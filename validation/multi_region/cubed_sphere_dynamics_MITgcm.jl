#=
Download:
(a) the file old_code_metrics.jld2 from
    https://www.dropbox.com/scl/fo/qu7nfr94wqc6ym6izpfqw/h?rlkey=zd4o5134u64ibyxggy64tiygt&dl=0; and
(b) the directory grid_cs32+ol4 from
    https://www.dropbox.com/scl/fo/c0pex0u8yvao6ehd3rqtp/h?rlkey=uq8bojrrsa7c4pb4n9ou8wvcs&dl=0;
and place them in the path validation/multi_region/. Then run this script from the same path as:
include("cubed_sphere_dynamics_MITgcm.jl")
=#

using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, xnode, ynode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halos_of_paired_fields!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_velocity_halos!
using Oceananigans.Operators
using Oceananigans.Utils: Iterate
using DataDeps
using JLD2
using CairoMakie
# to compute the vorticity:
using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

include("cubed_sphere_visualization.jl")

g = 10.

Lz = 1000
R  = 6370e3        # sphere's radius
U  = 40.           # velocity scale

# Solid body and planet rotation:
Ω_prime = U/R
π_MITgcm = 3.14159265358979323844
Ω = 2π_MITgcm/86400

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
                                    z = (-Lz, 0),
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
                                      z = (-Lz, 0),
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
            old_xᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/xC.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_xᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/xG.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_yᶜᶜᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/yC.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
            old_yᶠᶠᵃ_parent[:, :, region]  =  read_big_endian_coordinates("grid_cs32+ol4/yG.00$(region).001.data", 32, 4)[1+4-nHalo:end-4+nHalo,1+4-nHalo:end-4+nHalo]
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

my_Coriolis = HydrostaticSphericalCoriolis( rotation_rate = Ω ,
                                            scheme = EnstrophyConserving())

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(vorticity_scheme = EnergyConserving(),
                                                                         vertical_scheme = EnergyConserving()),
                                    free_surface = ExplicitFreeSurface(; gravitational_acceleration = g),
                                    coriolis = my_Coriolis,
                                    buoyancy = nothing)

# model.timestepper.χ = -0.5   # to switch to Forward Euler time-stepping (no AB-2)

# Define Solid body rotation flow field:

ψᵣ(λ, φ, z) = -R^2*Ω_prime*sind(φ) # ψᵣ(λ, φ, z) = -U * R * sind(φ)

#=
for φʳ = 90; ψᵣ(λ, φ, z) = - U * R * sind(φ)
             uᵣ(λ, φ, z) = - 1 / R * ∂φ(ψᵣ) = U * cosd(φ)
             vᵣ(λ, φ, z) = + 1 / (R * cosd(φ)) * ∂λ(ψᵣ) = 0
             ζᵣ(λ, φ, z) = - 1 / (R * cosd(φ)) * ∂φ(uᵣ * cosd(φ)) = 2 * (U / R) * sind(φ)
=#
ψ = Field{Face, Face, Center}(grid)

# Note that set! fills only interior points; to compute u and v we need information in the halo regions.
set!(ψ, ψᵣ)

#=
Note: fill_halo_regions! works for (Face, Face, Center) field, *except* for the two corner points that do not correspond
to an interior point! We need to manually fill the Face-Face halo points of the two corners that do not have a
corresponding interior point.
=#

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

for passes in 1:3
    fill_halo_regions!(ψ)
end

#=
for region in 1:number_of_regions(grid)
    for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in 1:grid.Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
        #=
        At the halo points, both latitude (φ) and longitude (λ) assume zero values, which in turn causes the
        streamfunction (ψ) to be zero. Therefore, to guarantee accurate values at these points, we opted to reinstate
        the fill halos for the streamfunction (ψ) after setting its interior values.
        =#
    end
end
=#

u = XFaceField(grid)
v = YFaceField(grid)

#=
c₁ = 1
c₂ = 2
=#

for region in 1:number_of_regions(grid)
    for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
        #=
        # Debugging option 1
        u[region][i, j, k] = 1
        v[region][i, j, k] = 2
        =#
        #=
        # Debugging option 2
        x = xnode(i, j, k, grid[region], Center(), Face(), Center())
        y = ynode(i, j, k, grid[region], Face(), Center(), Center())
        u[region][i, j, k] = c₁ * y
        v[region][i, j, k] = c₂ * x
        # Specify v[region][i, j, k] = 0 for further debugging and turn off visualization of the initial meridional velocity field.
        =#
        u[region][i, j, k] = - (ψ[region][i, j+1, k] - ψ[region][i, j, k]) / grid[region].Δyᶠᶜᵃ[i, j]
        v[region][i, j, k] =   (ψ[region][i+1, j, k] - ψ[region][i, j, k]) / grid[region].Δxᶜᶠᵃ[i, j]
    end
end

#=
fill_velocity_halos!((; u, v, w = nothing))
=#
fill_halos_of_paired_fields!((u, v))

# Now, compute the vorticity.

ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

#=
Upon examining the initial vorticity field plot, it was noted that NANs unexpectedly appear along the halos.
Additionally, the vorticity values along the boundaries are significantly higher compared to those within the domain's
interior. These issues likely contribute to the instability in the solution. By replacing the line
@inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
with
@inbounds ζ[i, j, k] = (-1)^(i+j+k)*(i + j + k)
we observe that all points are populated with finite values, effectively eliminating the appearance of NANs.
Consequently, it's evident that the flaw lies within the vorticity computation function, which seems to be producing
erroneous results.
=#

offset = -1 .* halo_size(grid)

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
end

# set Initial conditions

fac = -(R^2) * Ω_prime * (Ω + 0.5Ω_prime) / g

for region in 1:number_of_regions(grid)
    #=
    The following operations only set the interior values of the model velocities.
    model.velocities.u[region] .= u[region]
    model.velocities.v[region] .= v[region]
    =#

    for j in 1-Hy:grid.Ny+Hy, i in 1-Hx:grid.Nx+Hx, k in 1:grid.Nz
        model.velocities.u[region][i,j,k] = u[region][i, j, k]
        model.velocities.v[region][i,j,k] = v[region][i, j, k]
    end

    for j in 1:grid.Ny, i in 1:grid.Nx, k in grid.Nz+1:grid.Nz+1
        φ = φnode(i, j, k, grid[region], Center(), Center(), Center())
        model.free_surface.η[region][i, j, k] = fac * ( (sind(φ))^2 -1/3 )
    end
end

for passes in 1:3
    fill_halo_regions!(model.free_surface.η)
end

Δt =  600
stop_time = 10*86400 # 10 days, close to revolution period = 11.58 days
simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(6))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

eta_fields = Field[]
save_eta(sim) = push!(eta_fields, deepcopy(sim.model.free_surface.η))

ζ = Field{Face, Face, Center}(grid)

ζ_fields = Field[]
Δζ_fields = Field[]

@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
end

u₀ = deepcopy(simulation.model.velocities.u)
v₀ = deepcopy(simulation.model.velocities.v)
ζ₀ = deepcopy(ζ)

# Plot the relative difference of the grid metrics with halos.

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

# Plot the initial velocity field after model definition.

fig = panel_wise_visualization_with_halos(grid, u₀)
save("u₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, u₀)
save("u₀.png", fig)

fig = panel_wise_visualization_with_halos(grid, v₀)
save("v₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, v₀)
save("v₀.png", fig)

# Plot the initial surface elevation field after model definition.

fig = panel_wise_visualization_with_halos(grid, simulation.model.free_surface.η, grid.Nz+1, true, true)
save("η₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, simulation.model.free_surface.η, grid.Nz+1, true, true)
save("η₀.png", fig)

# Plot the initial vorticity field after model definition.

fig = panel_wise_visualization_with_halos(grid, ζ₀)
save("ζ₀_with_halos.png", fig)

fig = panel_wise_visualization(grid, ζ₀)
save("ζ₀.png", fig)

function save_vorticity(sim)
    Hx, Hy, Hz = halo_size(grid)

    #=
    fill_velocity_halos!(sim.model.velocities)
    =#
    fill_halos_of_paired_fields!((sim.model.velocities.u, sim.model.velocities.v))

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
            Δζ_field[region][i, j, k] -= ζ₀[region][i, j, k]
        end
    end

    push!(Δζ_fields, Δζ_field)
end

save_fields_iteration_interval = 12
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_eta] = Callback(save_eta, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(save_fields_iteration_interval))

run!(simulation)

#=
fig = panel_wise_visualization(grid, Δζ_fields[end])
save("Δζ.png", fig)

fig = panel_wise_visualization(grid, ζ₀)
save("ζ₀.png", fig)

start_index = 1
use_symmetric_colorrange = true
animation_time = 10 # seconds
framerate = floor(Int, size(Δζ_fields)[1]/animation_time)

create_panel_wise_visualization_animation(grid, Δζ_fields, start_index, use_symmetric_colorrange, framerate, "Δζ")
=#

using Oceananigans, JLD2, OffsetArrays, Statistics

# Comparison of 32x32 cubed sphere grid coordinates and metrics relative to their counterparts from MITgcm

Nx, Ny, Nz = 32, 32, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=6370e3, horizontal_direction_halo = 4,
                                  z_halo = 1)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

MITgcm_cs_grid_file = jldopen("cubed_sphere_32_grid_with_4_halos.jld2")

λᶜᶜᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
λᶠᶠᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶜᶜᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶠᶠᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)

jldopen("MITgcm_cs_grid_relative_difference.jld2", "w") do file
    for region in 1:6
         λᶜᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].λᶜᶜᵃ  - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].λᶜᶜᵃ  .+ 100eps(eltype(grid)))
         λᶠᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].λᶠᶠᵃ  - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].λᶠᶠᵃ  .+ 100eps(eltype(grid)))
         φᶜᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].φᶜᶜᵃ  - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].φᶜᶜᵃ  .+ 100eps(eltype(grid)))
         φᶠᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].φᶠᶠᵃ  - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].φᶠᶠᵃ  .+ 100eps(eltype(grid)))
        Δxᶠᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].Δxᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Δxᶠᶜᵃ .+ 100eps(eltype(grid)))
        Δxᶜᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].Δxᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Δxᶜᶠᵃ .+ 100eps(eltype(grid)))
        Δyᶠᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].Δyᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Δyᶠᶜᵃ .+ 100eps(eltype(grid)))
        Δyᶜᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].Δyᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Δyᶜᶠᵃ .+ 100eps(eltype(grid)))
        Azᶜᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].Azᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Azᶜᶜᵃ .+ 100eps(eltype(grid)))
        Azᶠᶜᵃ_difference_MITgcm[:, :, region] = (grid[region].Azᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Azᶠᶜᵃ .+ 100eps(eltype(grid)))
        Azᶜᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].Azᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Azᶜᶠᵃ .+ 100eps(eltype(grid)))
        Azᶠᶠᵃ_difference_MITgcm[:, :, region] = (grid[region].Azᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)) ./ (grid[region].Azᶠᶠᵃ .+ 100eps(eltype(grid)))
        file["λᶜᶜᵃ_relative_difference_MITgcm/" * string(region)]  =  λᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["λᶠᶠᵃ_relative_difference_MITgcm/" * string(region)]  =  λᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["φᶜᶜᵃ_relative_difference_MITgcm/" * string(region)]  =  φᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["φᶠᶠᵃ_relative_difference_MITgcm/" * string(region)]  =  φᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Δxᶠᶜᵃ_relative_difference_MITgcm/" * string(region)] = Δxᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δxᶜᶠᵃ_relative_difference_MITgcm/" * string(region)] = Δxᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Δyᶠᶜᵃ_relative_difference_MITgcm/" * string(region)] = Δyᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δyᶜᶠᵃ_relative_difference_MITgcm/" * string(region)] = Δyᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶜᵃ_relative_difference_MITgcm/" * string(region)] = Azᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶜᵃ_relative_difference_MITgcm/" * string(region)] = Azᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶠᵃ_relative_difference_MITgcm/" * string(region)] = Azᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶠᵃ_relative_difference_MITgcm/" * string(region)] = Azᶠᶠᵃ_difference_MITgcm[:, :, region]
    end
end

close(MITgcm_cs_grid_file)

function zero_out_corners!(array, (Nx, Ny), (Hx, Hy))
    array[1:Hx, 1:Hy] .= 0
    array[1:Hx, Ny+Hy+1:Ny+2Hy] .= 0
    array[Nx+Hx+1:Nx+2Hx, 1:Hy] .= 0
    array[Nx+Hx+1:Nx+2Hx, Ny+Hy+1:Ny+2Hy] .= 0
    return nothing
end

function variance_excluding_corners(array, (Nx, Ny), (Hx, Hy))
    zero_out_corners!(array, (Nx, Ny), (Hx, Hy))
    return mean(abs, array)
end

file = jldopen("MITgcm_cs_grid_relative_difference.jld2")

for varname in ("λᶜᶜᵃ",
                "λᶜᶜᵃ",
                "λᶠᶠᵃ",
                "φᶜᶜᵃ",
                "φᶠᶠᵃ",
                "Δxᶠᶜᵃ",
                "Δxᶜᶠᵃ",
                "Δyᶠᶜᵃ",
                "Δyᶜᶠᵃ",
                "Azᶜᶜᵃ",
                "Azᶠᶜᵃ",
                "Azᶜᶠᵃ",
                "Azᶠᶠᵃ")

    for region in 1:6
        array = deepcopy(file[varname * "_relative_difference_MITgcm/" * string(region)])
        relative_error = variance_excluding_corners(array, (32, 32), (4, 4))
        @info varname * " panel " * string(region) * ": " * string(relative_error)
    end
end

close(file)

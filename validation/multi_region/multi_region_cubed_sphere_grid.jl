using Oceananigans, JLD2, OffsetArrays

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

jldopen("MITgcm_cs_grid_difference.jld2", "w") do file
    for region in 1:6
        λᶜᶜᵃ_difference_MITgcm[:, :, region]  =  grid[region].λᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        λᶠᶠᵃ_difference_MITgcm[:, :, region]  =  grid[region].λᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶜᶜᵃ_difference_MITgcm[:, :, region]  =  grid[region].φᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶠᶠᵃ_difference_MITgcm[:, :, region]  =  grid[region].φᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Δxᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Δxᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Δyᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Δyᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        file["λᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  λᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["λᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  λᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["φᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  φᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["φᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  φᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Δxᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δxᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δxᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δxᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Δyᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δyᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δyᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δyᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶜᵃ_difference_MITgcm/" * string(region)] = Azᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶜᵃ_difference_MITgcm/" * string(region)] = Azᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶠᵃ_difference_MITgcm/" * string(region)] = Azᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶠᵃ_difference_MITgcm/" * string(region)] = Azᶠᶠᵃ_difference_MITgcm[:, :, region]
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

using Statistics

function variance_excluding_corners(array, (Nx, Ny), (Hx, Hy))
    zero_out_corners!(array, (Nx, Ny), (Hx, Hy))
    return mean(abs, array)
end

Nx = Ny = 32
Hx = Hy = 4

region = 1

file = jldopen("MITgcm_cs_grid_difference.jld2")

for varname in ("λᶜᶜᵃ_difference_MITgcm",
                "λᶜᶜᵃ_difference_MITgcm",
                "λᶠᶠᵃ_difference_MITgcm",
                "φᶜᶜᵃ_difference_MITgcm",
                "φᶠᶠᵃ_difference_MITgcm",
                "Δxᶠᶜᵃ_difference_MITgcm",
                "Δxᶜᶠᵃ_difference_MITgcm",
                "Δyᶠᶜᵃ_difference_MITgcm",
                "Δyᶜᶠᵃ_difference_MITgcm",
                "Azᶜᶜᵃ_difference_MITgcm",
                "Azᶠᶜᵃ_difference_MITgcm",
                "Azᶜᶠᵃ_difference_MITgcm",
                "Azᶠᶠᵃ_difference_MITgcm")

    for region in 1:6
        local array = deepcopy(file[varname * "/" * string(region)])
        error = variance_excluding_corners(array, (32, 32), (4, 4))
        @info varname * " panel " * string(region) * ": " * string(error)
    end
end

close(file)

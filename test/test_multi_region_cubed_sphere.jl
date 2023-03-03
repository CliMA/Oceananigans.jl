include("dependencies_for_runtests.jl")

using Oceananigans.Utils: Iterate
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.MultiRegion: getregion

# Adopted from figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
# The configuration of the panels for the cubed sphere. Each panel is partitioned in two parts YPartition(2)
#
#                              ponel P5      panel P6
#                           + ---------- + ---------- +
#                           |     ↑↑     |     ↑↑     |
#                           |     1W     |     1S     |
#                           |←3N      6W→|←5E      2S→|
#                           |------------|------------|
#                           |←3N      6W→|←5E      2S→|
#                           |     4N     |     4E     |
#                 panel P3  |     ↓↓     |     ↓↓     |
#              + ---------- +------------+------------+
#              |     ↑↑     |     ↑↑     | 
#              |     5W     |     5S     | 
#              |←1N      4W→|←3E      6S→| 
#              |------------|------------| 
#              |←1N      4W→|←3E      6S→| 
#              |     2N     |     2E     | 
#              |     ↓↓     |     ↓↓     | 
# + -----------+------------+------------+ 
# |     ↑↑     |     ↑↑     |  panel P4
# |     3W     |     3S     |
# |←5N      2W→|←1E      4S→|
# |------------|------------|
# |←5N      2W→|←1E      4S→|
# |     6N     |     6E     |
# |     ↓↓     |     ↓↓     |
# + -----------+------------+
#   panel P1   panel P2


    #                         panel P5   panel P6
    #                       +----------+----------+
    #                       |    ↑↑    |    ↑↑    |
    #                       |    1W    |    1S    |
    #                       |←3N P5 6W→|←5E P6 2S→|
    #                       |    4N    |    4E    |
    #              panel P3 |    ↓↓    |    ↓↓    |
    #            +----------+----------+----------+
    #            |    ↑↑    |    ↑↑    |
    #            |    5W    |    5S    |
    #            |←1N P3 4W→|←3E P4 6S→|
    #            |    2N    |    2E    |
    #            |    ↓↓    |    ↓↓    |
    # +----------+----------+----------+
    # |    ↑↑    |    ↑↑    | panel P4
    # |    3W    |    3S    |
    # |←5N P1 2W→|←1E P2 4S→|
    # |    6N    |    6E    |
    # |    ↓↓    |    ↓↓    |
    # +----------+----------+
    #   panel P1   panel P2


function get_halo_data(field, side, k_index=1)
    Nx, Ny, _ = size(field)

    if side == :west
        return field.data[0, 1:Ny, k_index]
    elseif side == :east
        return field.data[Nx+1, 1:Ny, k_index]
    elseif side == :south
        return field.data[1:Nx, 0, k_index]
    elseif side == :north
        return field.data[1:Nx, Ny+1, k_index]
    end
end

@testset "Conformal cubed sphere face grid from file" begin
    @info "  Testing conformal cubed sphere face grid construction from file..."

    Nz = 1
    z = (-1, 0)

    cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

    for panel in 1:6
        grid = OrthogonalSphericalShellGrid(cs32_filepath; panel, Nz, z)
        @test grid isa OrthogonalSphericalShellGrid
    end

    for arch in archs

        # read cs32 grid from file
        grid_cs32 = ConformalCubedSphereGrid(cs32_filepath, arch; Nz, z)

        Nx, Ny, Nz = size(grid_cs32.region_grids[1])
        radius = grid_cs32.region_grids[1].radius

        # construct a ConformalCubedSphereGrid similar to cs32
        grid = ConformalCubedSphereGrid(arch; z, panel_size=(Nx, Ny, Nz), radius)

        for panel in 1:6
            # we test on cca and ffa; fca and cfa are all zeros on grid_cs32!
            @test isapprox(grid.region_grids[panel].φᶜᶜᵃ, grid_cs32.region_grids[panel].φᶜᶜᵃ)
            @test isapprox(grid.region_grids[panel].λᶜᶜᵃ, grid_cs32.region_grids[panel].λᶜᶜᵃ)

            # before we test, make sure we don't consider +180 and -180 longitudes as being "different"
            grid.region_grids[panel].λᶠᶠᵃ[grid.region_grids[panel].λᶠᶠᵃ .≈ -180] .= 180

            # and if poles are included, they have the same longitude
            grid.region_grids[panel].λᶠᶠᵃ[grid.region_grids[panel].φᶠᶠᵃ .≈ +90] = grid_cs32.region_grids[panel].λᶠᶠᵃ[grid.region_grids[panel].φᶠᶠᵃ .≈ +90]
            grid.region_grids[panel].λᶠᶠᵃ[grid.region_grids[panel].φᶠᶠᵃ .≈ -90] = grid_cs32.region_grids[panel].λᶠᶠᵃ[grid.region_grids[panel].φᶠᶠᵃ .≈ -90]
            @test isapprox(grid.region_grids[panel].φᶠᶠᵃ, grid_cs32.region_grids[panel].φᶠᶠᵃ)
            @test isapprox(grid.region_grids[panel].λᶠᶠᵃ, grid_cs32.region_grids[panel].λᶠᶠᵃ)
        end
    end
end


for FT in float_types
    for arch in archs
        Nx, Ny, Nz = 10, 10, 1

        grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1.0)

        c = CenterField(grid)

        regions = Iterate(Tuple(i for i in 1:length(grid)))

        set!(c, regions)

        fill_halo_regions!(c)

        @testset "Testing conformal cubed sphere grid halo filling for a tracer field" begin

            @info "  Testing halo filling for tracer [$FT, $(typeof(arch))]..."

            @test get_halo_data(getregion(c, 1), :west)  == 5 * ones(Ny)
            @test get_halo_data(getregion(c, 1), :east)  == 2 * ones(Ny)
            @test get_halo_data(getregion(c, 1), :south) == 6 * ones(Ny)
            @test get_halo_data(getregion(c, 1), :north) == 3 * ones(Ny)

            @test get_halo_data(getregion(c, 2), :west)  == 1 * ones(Ny)
            @test get_halo_data(getregion(c, 2), :east)  == 4 * ones(Ny)
            @test get_halo_data(getregion(c, 2), :south) == 6 * ones(Ny)
            @test get_halo_data(getregion(c, 2), :north) == 3 * ones(Ny)

            @test get_halo_data(getregion(c, 3), :west)  == 1 * ones(Ny)
            @test get_halo_data(getregion(c, 3), :east)  == 4 * ones(Ny)
            @test get_halo_data(getregion(c, 3), :south) == 2 * ones(Ny)
            @test get_halo_data(getregion(c, 3), :north) == 5 * ones(Ny)

            @test get_halo_data(getregion(c, 4), :west)  == 3 * ones(Ny)
            @test get_halo_data(getregion(c, 4), :east)  == 6 * ones(Ny)
            @test get_halo_data(getregion(c, 4), :south) == 2 * ones(Ny)
            @test get_halo_data(getregion(c, 4), :north) == 5 * ones(Ny)

            @test get_halo_data(getregion(c, 5), :west)  == 3 * ones(Ny)
            @test get_halo_data(getregion(c, 5), :east)  == 6 * ones(Ny)
            @test get_halo_data(getregion(c, 5), :south) == 4 * ones(Ny)
            @test get_halo_data(getregion(c, 5), :north) == 1 * ones(Ny)

            @test get_halo_data(getregion(c, 6), :west)  == 5 * ones(Ny)
            @test get_halo_data(getregion(c, 6), :east)  == 2 * ones(Ny)
            @test get_halo_data(getregion(c, 6), :south) == 4 * ones(Ny)
            @test get_halo_data(getregion(c, 6), :north) == 1 * ones(Ny)
        end
    end
end

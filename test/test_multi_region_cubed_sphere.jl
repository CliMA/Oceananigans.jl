include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Utils: Iterate
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.MultiRegion: getregion

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

@testset "Testing conformal cubed sphere partitions..." begin
    for n = 1:4
        @test length(CubedSpherePartition(; R=n)) == 6n^2
    end
end

@testset "Testing conformal cubed sphere face grid from file" begin
    Nz = 1
    z = (-1, 0)

    cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

    for panel in 1:6
        grid = OrthogonalSphericalShellGrid(cs32_filepath; panel, Nz, z)
        @test grid isa OrthogonalSphericalShellGrid
    end

    for arch in archs
        @info "  Testing conformal cubed sphere face grid from file [$(typeof(arch))]..."

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

@testset "Testing conformal cubed sphere fill halos for tracers" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos for tracers [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 10, 10, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1.0)

            c = CenterField(grid)

            # fill the tracer with values = panel_index
            regions = Iterate(Tuple(i for i in 1:length(grid)))
            set!(c, regions)

            fill_halo_regions!(c)

            # confirm that the halos were filled according to connectivity
            # described at ConformalCubedSphereGrid docstring

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

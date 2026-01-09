include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.OrthogonalSphericalShellGrids: ConformalCubedSpherePanelGrid
using Oceananigans.Utils: Iterate, getregion
using Oceananigans.MultiRegion: number_of_regions, fill_halo_regions!

function get_range_of_indices(operation, index, Nx, Ny)
    if operation == :endpoint && index == :before_first
        range_x = 0
        range_y = 0
    elseif operation == :endpoint && index == :first
        range_x = 1
        range_y = 1
    elseif operation == :endpoint && index == :last
        range_x = Nx
        range_y = Ny
    elseif operation == :endpoint && index == :after_last
        range_x = Nx + 1
        range_y = Ny + 1
    elseif operation == :subset && index == :first # here index is the index to skip
        range_x = 2:Nx
        range_y = 2:Ny
    elseif operation == :subset && index == :last # here index is the index to skip
        range_x = 1:Nx-1
        range_y = 1:Ny-1
    else
        range_x = 1:Nx
        range_y = 1:Ny
    end

    return range_x, range_y
end

function get_halo_data(field, ::West, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[-Hx+1:0, range_y, k_index]
end

function get_halo_data(field, ::East, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[Nx+1:Nx+Hx, range_y, k_index]
end

function get_halo_data(field, ::North, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[range_x, Ny+1:Ny+Hy, k_index]
end

function get_halo_data(field, ::South, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[range_x, -Hy+1:0, k_index]
end

function get_boundary_indices(Nx, Ny, Hx, Hy, ::West; operation=nothing, index=:all)
    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return 1:Hx, range_y
end

function get_boundary_indices(Nx, Ny, Hx, Hy, ::South; operation=nothing, index=:all)
    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)

    return range_x, 1:Hy
end

function get_boundary_indices(Nx, Ny, Hx, Hy, ::East; operation=nothing, index=:all)
    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return Nx-Hx+1:Nx, range_y
end

function get_boundary_indices(Nx, Ny, Hx, Hy, ::North; operation=nothing, index=:all)
    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)

    return range_x, Ny-Hy+1:Ny
end

# Solid body rotation
R = 1        # sphere's radius
U = 1        # velocity scale
φʳ = 0       # Latitude pierced by the axis of rotation
α  = 90 - φʳ # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

"""
    create_test_data(grid, region)

Create an array with integer values of the form, e.g., 541 corresponding to region=5, i=4, j=2. If `trailing_zeros > 0`
then all values are multiplied with `10^trailing_zeros`, e.g., for `trailing_zeros = 2` we have that 54100 corresponds
to region=5, i=4, j=2.
"""
function create_test_data(grid, region; trailing_zeros=0)
    Nx, Ny, Nz = size(grid)
    (Nx > 9 || Ny > 9) && error("you provided (Nx, Ny) = ($Nx, $Ny); use a grid with Nx, Ny ≤ 9.")
    !(trailing_zeros isa Integer) && error("trailing_zeros has to be an integer")
    factor = 10^(trailing_zeros)

    return factor .* [100region + 10i + j for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

function fill_missing_corners!(grid, region, test_data)
    Nx, Ny, _ = size(grid)
    if isodd(region)
        test_data[1, Ny+1, :] .= 0
    else
        test_data[Nx+1, 1, :] .= 0
    end
end

create_c_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=0)

function create_ψ_test_data(grid, region)
    ψ_test_data = create_test_data(grid, region; trailing_zeros=1)
    fill_missing_corners!(grid, region, ψ_test_data)
    return ψ_test_data
end

create_c₁_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=2)
create_c₂_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=3)

create_u_test_data(grid, region)  = create_test_data(grid, region; trailing_zeros=4)
create_v_test_data(grid, region)  = create_test_data(grid, region; trailing_zeros=5)

function create_ψ₁_test_data(grid, region)
    ψ₁_test_data = create_test_data(grid, region; trailing_zeros=6)
    fill_missing_corners!(grid, region, ψ₁_test_data)
    ψ₁_test_data[1, 1, :] .= 0
    return ψ₁_test_data
end

function create_ψ₂_test_data(grid, region)
    ψ₂_test_data = create_test_data(grid, region; trailing_zeros=7)
    fill_missing_corners!(grid, region, ψ₂_test_data)
    ψ₂_test_data[1, 1, :] .= 0
    return ψ₂_test_data
end

"""
    same_longitude_at_poles!(grid_1, grid_2)

Change the longitude values in `grid_1` that correspond to points situated _exactly_
at the poles so that they match the corresponding longitude values of `grid_2`.
"""
function same_longitude_at_poles!(grid_1::ConformalCubedSphereGrid, grid_2::ConformalCubedSphereGrid)
    number_of_regions(grid_1) == number_of_regions(grid_2) || error("grid_1 and grid_2 must have same number of regions")

    for region in 1:number_of_regions(grid_1)
        grid_1[region].λᶠᶠᵃ[grid_2[region].φᶠᶠᵃ .== +90]= grid_2[region].λᶠᶠᵃ[grid_2[region].φᶠᶠᵃ .== +90]
        grid_1[region].λᶠᶠᵃ[grid_2[region].φᶠᶠᵃ .== -90]= grid_2[region].λᶠᶠᵃ[grid_2[region].φᶠᶠᵃ .== -90]
    end

    return nothing
end

"""
    zero_out_corner_halos!(array::OffsetArray, N, H)

Zero out the values at the corner halo regions of the two-dimensional `array`.
It is expected that the interior of the offset `array` is `(Nx, Ny) = (N, N)` and
the halo region is `H` in both dimensions.
"""
function zero_out_corner_halos!(array::OffsetArray, N, H)
    size(array) == (N+2H, N+2H)

    Nx = Ny = N
    Hx = Hy = H

    array[-Hx+1:0, -Hy+1:0] .= 0
    array[-Hx+1:0, Ny+1:Ny+Hy] .= 0
    array[Nx+1:Nx+Hx, -Hy+1:0] .= 0
    array[Nx+1:Nx+Hx, Ny+1:Ny+Hy] .= 0

    return nothing
end

function compare_grid_vars(var1, var2, N, H)
    zero_out_corner_halos!(var1, N, H)
    zero_out_corner_halos!(var2, N, H)
    return isapprox(var1, var2)
end

@testset "Testing conformal cubed sphere partitions..." begin
    for n = 1:4
        @test length(CubedSpherePartition(; R=n)) == 6n^2
    end
end

@testset "Testing conformal cubed sphere grid from file" begin
    Nz = 1
    z = (-1, 0)

    cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid_with_4_halos.jld2"

    for panel in 1:6
        grid = ConformalCubedSpherePanelGrid(cs32_filepath; panel, Nz, z)
        @test grid isa OrthogonalSphericalShellGrid
    end

    for arch in archs
        @info "  Testing conformal cubed sphere grid from file [$(typeof(arch))]..."

        # Read cs32 grid from file.
        grid_cs32 = ConformalCubedSphereGrid(cs32_filepath, arch; Nz, z)

        radius = first(grid_cs32).radius
        Nx, Ny, Nz = size(grid_cs32)
        Hx, Hy, Hz = halo_size(grid_cs32)

        Nx !== Ny && error("Nx must be same as Ny")
        N = Nx
        Hx !== Hy && error("Hx must be same as Hy")
        H = Hy

        # Construct a ConformalCubedSphereGrid similar to cs32.
        grid = ConformalCubedSphereGrid(arch; z, panel_size=(Nx, Ny, Nz), radius,
                                        horizontal_direction_halo = Hx, z_halo = Hz)

        for panel in 1:6
            @allowscalar begin
                # Test only on cca and ffa; fca and cfa are all zeros on grid_cs32!
                # Only test interior points since halo regions are not filled for grid_cs32!

                @test compare_grid_vars(getregion(grid, panel).φᶜᶜᵃ, getregion(grid_cs32, panel).φᶜᶜᵃ, N, H)
                @test compare_grid_vars(getregion(grid, panel).λᶜᶜᵃ, getregion(grid_cs32, panel).λᶜᶜᵃ, N, H)

                # Before we test, make sure we don't consider +180 and -180 longitudes as being "different".
                getregion(grid, panel).λᶠᶠᵃ[getregion(grid, panel).λᶠᶠᵃ .≈ -180] .= 180

                # If poles are included, they have the same longitude.
                same_longitude_at_poles!(grid, grid_cs32)

                @test compare_grid_vars(getregion(grid, panel).φᶠᶠᵃ, getregion(grid_cs32, panel).φᶠᶠᵃ, N, H)
                @test compare_grid_vars(getregion(grid, panel).λᶠᶠᵃ, getregion(grid_cs32, panel).λᶠᶠᵃ, N, H)

                @test compare_grid_vars(getregion(grid, panel).φᶠᶠᵃ, getregion(grid_cs32, panel).φᶠᶠᵃ, N, H)
                @test compare_grid_vars(getregion(grid, panel).λᶠᶠᵃ, getregion(grid_cs32, panel).λᶠᶠᵃ, N, H)
            end
        end
    end
end

panel_sizes = ((8, 8, 1), (9, 9, 2))

@testset "Testing area metrics" begin
    for FT in float_types, arch in archs, panel_size in panel_sizes, non_uniform_conformal_mapping in (false, true)
        Nx, Ny, Nz = panel_size
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"

        @info "  Testing conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."

        grid = ConformalCubedSphereGrid(arch, FT;
                                        panel_size, z = (0, 1), radius = 1, non_uniform_conformal_mapping)

        areaᶜᶜᵃ = areaᶠᶜᵃ = areaᶜᶠᵃ = areaᶠᶠᵃ = 0

        for region in 1:number_of_regions(grid)
            @allowscalar begin
                areaᶜᶜᵃ += sum(getregion(grid, region).Azᶜᶜᵃ[1:Nx, 1:Ny])
                areaᶠᶜᵃ += sum(getregion(grid, region).Azᶠᶜᵃ[1:Nx, 1:Ny])
                areaᶜᶠᵃ += sum(getregion(grid, region).Azᶜᶠᵃ[1:Nx, 1:Ny])
                areaᶠᶠᵃ += sum(getregion(grid, region).Azᶠᶠᵃ[1:Nx, 1:Ny])
            end
        end

        @test areaᶜᶜᵃ ≈ areaᶠᶜᵃ ≈ areaᶜᶠᵃ ≈ areaᶠᶠᵃ ≈ 4π * grid.radius^2
    end
end

@testset "Immersed conformal cubed sphere construction" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        Nx, Ny, Nz = 9, 9, 9
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"

        @info "  Testing immersed conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (-1, 0), radius = 1,
                                                   non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        # Test that the grid is constructed correctly.
        for panel in 1:6
            grid = getregion(immersed_grid, panel)

            if panel == 3 || panel == 6 # North and South panels should be completely immersed.
                @test isempty(grid.interior_active_cells)
            else # Other panels should have some active cells.
                @test !isempty(grid.interior_active_cells)
            end
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for center fields such as tracers" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        @info "  Testing fill halos for center fields [$FT, $(typeof(arch)), $cm]..."

        Nx, Ny, Nz = 9, 9, 1

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                   horizontal_direction_halo = 3, non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        grids = (underlying_grid, immersed_grid)

        for grid in grids
            c = CenterField(grid)

            region = Iterate(1:6)
            @apply_regionally data = create_c_test_data(grid, region)
            set!(c, data)

            fill_halo_regions!(c)

            Hx, Hy, Hz = halo_size(grid)

            west_indices  = 1:Hx, 1:Ny
            south_indices = 1:Nx, 1:Hy
            east_indices  = Nx-Hx+1:Nx, 1:Ny
            north_indices = 1:Nx, Ny-Hy+1:Ny

            # Confirm that the tracer halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            @allowscalar begin
                for panel in 1:6
                    west_panel = grid.connectivity.connections[panel].west.from_rank
                    east_panel = grid.connectivity.connections[panel].east.from_rank
                    south_panel = grid.connectivity.connections[panel].south.from_rank
                    north_panel = grid.connectivity.connections[panel].north.from_rank

                    if isodd(panel)
                        @test get_halo_data(getregion(c, panel), West())  == reverse(create_c_test_data(grid, west_panel)[north_indices...], dims=1)'
                        @test get_halo_data(getregion(c, panel), East())  ==         create_c_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(c, panel), South()) ==         create_c_test_data(grid, south_panel)[north_indices...]
                        @test get_halo_data(getregion(c, panel), North()) == reverse(create_c_test_data(grid, north_panel)[west_indices...], dims=2)'
                    else
                        @test get_halo_data(getregion(c, panel), West())  ==         create_c_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(c, panel), East())  == reverse(create_c_test_data(grid, east_panel)[south_indices...], dims=1)'
                        @test get_halo_data(getregion(c, panel), South()) == reverse(create_c_test_data(grid, south_panel)[east_indices...], dims=2)'
                        @test get_halo_data(getregion(c, panel), North()) ==         create_c_test_data(grid, north_panel)[south_indices...]
                    end
                end
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for face-face-any fields such as streamfunction" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        @info "  Testing fill halos for face-face-any fields [$FT, $(typeof(arch)), $cm]..."

        Nx, Ny, Nz = 9, 9, 1

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                   horizontal_direction_halo = 3, non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        grids = (underlying_grid, immersed_grid)

        for grid in grids
            ψ = Field{Face, Face, Center}(grid)

            region = Iterate(1:6)
            @apply_regionally data = create_ψ_test_data(grid, region)
            set!(ψ, data)

            fill_halo_regions!(ψ)

            Hx, Hy, Hz = halo_size(grid)

            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=nothing, index=:all) # (1:Hx, 1:Ny)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=nothing, index=:all) # (Nx-Hx+1:Nx, 1:Ny)
            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=nothing, index=:all) # (1:Nx, 1:Hy)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=nothing, index=:all) # (1:Nx, Ny-Hy+1:Ny)

            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:first) # (1:Hx, 1)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:first) # (Nx-Hx+1:Nx, 1)
            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:first) # (1, 1:Hy)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:first) # (1, Ny-Hy+1:Ny)

            west_indices_first_shifted_east = first(west_indices_first[1]) + 1 : last(west_indices_first[1]) + 1, west_indices_first[2] # (2:Hx+1, 1)
            east_indices_first_shifted_east = first(east_indices_first[1]) + 1 : last(east_indices_first[1]) + 1, east_indices_first[2] # (Nx-Hx+2:Nx+1, 1)
            north_indices_first_shifted_north = north_indices_first[1], first(north_indices_first[2]) + 1 : last(north_indices_first[2]) + 1 # (1, Ny-Hy+2:Ny+1)
            south_indices_first_shifted_north = south_indices_first[1], first(south_indices_first[2]) + 1 : last(south_indices_first[2]) + 1 # (1, 2:Hy+1)

            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:first) # (1:Hx, 2:Ny)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:first) # (Nx-Hx+1:Nx, 2:Ny)
            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:first) # (2:Nx, 1:Hy)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:first) # (2:Nx, Ny-Hy+1:Ny)

            @allowscalar begin
                for panel in 1:6
                    west_panel = grid.connectivity.connections[panel].west.from_rank
                    east_panel = grid.connectivity.connections[panel].east.from_rank
                    south_panel = grid.connectivity.connections[panel].south.from_rank
                    north_panel = grid.connectivity.connections[panel].north.from_rank

                    if isodd(panel)
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ, panel), East())  ==         create_ψ_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(ψ, panel), South()) ==         create_ψ_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ, panel), West();
                                            operation=:endpoint,
                                            index=:first)                 ==         create_ψ_test_data(grid, south_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), West();
                                            operation=:subset,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, west_panel)[north_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, west_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, north_panel)[west_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)            == reverse(create_ψ_test_data(grid, east_panel)[west_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ, panel), North();
                                            operation=:endpoint,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, west_panel)[north_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ, panel), North();
                                            operation=:subset,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, north_panel)[west_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, north_panel)[west_indices_first...]
                    else
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ, panel), West())  ==         create_ψ_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(ψ, panel), North()) ==         create_ψ_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)            == reverse(create_ψ_test_data(grid, north_panel)[south_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ, panel), East();
                                            operation=:endpoint,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, south_panel)[east_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ, panel), East();
                                            operation=:subset,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, east_panel)[south_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, east_panel)[south_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), South();
                                            operation=:endpoint,
                                            index=:first)                 ==         create_ψ_test_data(grid, west_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), South();
                                            operation=:subset,
                                            index=:first)                 == reverse(create_ψ_test_data(grid, south_panel)[east_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, south_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)            ==         create_ψ_test_data(grid, east_panel)[south_indices_first...]
                    end
                end
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for center-center field pairs such as (Δxᶜᶜᵃ, Δyᶜᶜᵃ)" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        @info "  Testing fill halos for center-center field pairs [$FT, $(typeof(arch)), $cm]..."

        Nx, Ny, Nz = 9, 9, 1

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                   horizontal_direction_halo = 3, non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        grids = (underlying_grid, immersed_grid)

        for grid in grids
            c₁ = CenterField(grid)
            c₂ = CenterField(grid)

            region = Iterate(1:6)
            @apply_regionally c₁_data = create_c₁_test_data(grid, region)
            @apply_regionally c₂_data = create_c₂_test_data(grid, region)
            set!(c₁, c₁_data)
            set!(c₂, c₂_data)

            fill_halo_regions!((c₁, c₂))

            Hx, Hy, Hz = halo_size(grid)

            west_indices  = 1:Hx, 1:Ny
            south_indices = 1:Nx, 1:Hy
            east_indices  = Nx-Hx+1:Nx, 1:Ny
            north_indices = 1:Nx, Ny-Hy+1:Ny

            @allowscalar begin
                for panel in 1:6
                    west_panel = grid.connectivity.connections[panel].west.from_rank
                    east_panel = grid.connectivity.connections[panel].east.from_rank
                    south_panel = grid.connectivity.connections[panel].south.from_rank
                    north_panel = grid.connectivity.connections[panel].north.from_rank

                    if isodd(panel)
                        # Confirm that the c₁ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(c₁, panel), East())  ==           create_c₁_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(c₁, panel), South()) ==           create_c₁_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(c₁, panel), West())  ==   reverse(create_c₂_test_data(grid, west_panel)[north_indices...], dims=1)'
                        @test get_halo_data(getregion(c₁, panel), North()) == - reverse(create_c₂_test_data(grid, north_panel)[west_indices...], dims=2)'

                        # Confirm that the c₂ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(c₂, panel), East())  ==           create_c₂_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(c₂, panel), South()) ==           create_c₂_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(c₂, panel), West())  == - reverse(create_c₁_test_data(grid, west_panel)[north_indices...], dims=1)'
                        @test get_halo_data(getregion(c₂, panel), North()) ==   reverse(create_c₁_test_data(grid, north_panel)[west_indices...], dims=2)'
                    else
                        # Confirm that the c₁ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(c₁, panel), West())  ==           create_c₁_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(c₁, panel), North()) ==           create_c₁_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(c₁, panel), East())  ==   reverse(create_c₂_test_data(grid, east_panel)[south_indices...], dims=1)'
                        @test get_halo_data(getregion(c₁, panel), South()) == - reverse(create_c₂_test_data(grid, south_panel)[east_indices...], dims=2)'

                        # Confirm that the c₂ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(c₂, panel), West())  ==           create_c₂_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(c₂, panel), North()) ==           create_c₂_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(c₂, panel), East())  == - reverse(create_c₁_test_data(grid, east_panel)[south_indices...], dims=1)'
                        @test get_halo_data(getregion(c₂, panel), South()) ==   reverse(create_c₁_test_data(grid, south_panel)[east_indices...], dims=2)'
                    end
                end
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for face-center and center-face field pairs such as horizontal velocities" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        @info "  Testing fill halos for face-center and center-face field pairs [$FT, $(typeof(arch)), $cm]..."

        Nx, Ny, Nz = 9, 9, 1

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                   horizontal_direction_halo = 3, non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        grids = (underlying_grid, immersed_grid)

        for grid in grids
            u = XFaceField(grid)
            v = YFaceField(grid)

            region = Iterate(1:6)
            @apply_regionally u_data = create_u_test_data(grid, region)
            @apply_regionally v_data = create_v_test_data(grid, region)
            set!(u, u_data)
            set!(v, v_data)

            fill_halo_regions!((u, v))

            Hx, Hy, Hz = halo_size(grid)

            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=nothing, index=:all) # (1:Hx, 1:Ny)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=nothing, index=:all) # (Nx-Hx+1:Nx, 1:Ny)
            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=nothing, index=:all) # (1:Nx, 1:Hy)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=nothing, index=:all) # (1:Nx, Ny-Hy+1:Ny)

            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:first) # (1:Hx, 1)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:first) # (Nx-Hx+1:Nx, 1)
            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:first) # (1, 1:Hy)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:first) # (1, Ny-Hy+1:Ny)

            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:first) # (1:Hx, 2:Ny)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:first) # (Nx-Hx+1:Nx, 2:Ny)
            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:first) # (2:Nx, 1:Hy)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:first) # (2:Nx, Ny-Hy+1:Ny)

            @allowscalar begin
                for panel in 1:6
                    west_panel = grid.connectivity.connections[panel].west.from_rank
                    east_panel = grid.connectivity.connections[panel].east.from_rank
                    south_panel = grid.connectivity.connections[panel].south.from_rank
                    north_panel = grid.connectivity.connections[panel].north.from_rank

                    if isodd(panel)
                        # Confirm that the zonal velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(u, panel), East())  ==           create_u_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(u, panel), South()) ==           create_u_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(u, panel), West())  ==   reverse(create_v_test_data(grid, west_panel)[north_indices...], dims=1)'
                        @test get_halo_data(getregion(u, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)            ==   reverse(create_v_test_data(grid, east_panel)[west_indices_first...])
                        @test get_halo_data(getregion(u, panel), North();
                                            operation=:endpoint,
                                            index=:first)                 == - reverse(create_u_test_data(grid, west_panel)[north_indices_first...])
                        @test get_halo_data(getregion(u, panel), North();
                                            operation=:subset,
                                            index=:first)                 == - reverse(create_v_test_data(grid, north_panel)[west_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(u, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)            ==         - create_v_test_data(grid, north_panel)[west_indices_first...]

                        # Confirm that the meridional velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(v, panel), East())  ==           create_v_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(v, panel), South()) ==           create_v_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(v, panel), West();
                                            operation=:endpoint,
                                            index=:first)                 ==         - create_u_test_data(grid, south_panel)[north_indices_first...]
                        @test get_halo_data(getregion(v, panel), West();
                                            operation=:subset,
                                            index=:first)                 == - reverse(create_u_test_data(grid, west_panel)[north_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(v, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)            ==         - create_u_test_data(grid, west_panel)[north_indices_first...]
                        @test get_halo_data(getregion(v, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)            ==           create_v_test_data(grid, north_panel)[west_indices_first...]
                        @test get_halo_data(getregion(v, panel), North()) ==   reverse(create_u_test_data(grid, north_panel)[west_indices...], dims=2)'
                    else
                        # Confirm that the zonal velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(u, panel), West())  ==           create_u_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(u, panel), North()) ==           create_u_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(u, panel), East())  ==   reverse(create_v_test_data(grid, east_panel)[south_indices...], dims=1)'
                        @test get_halo_data(getregion(u, panel), South();
                                            operation=:endpoint,
                                            index=:first)                 ==         - create_v_test_data(grid, west_panel)[east_indices_first...]
                        @test get_halo_data(getregion(u, panel), South();
                                            operation=:subset,
                                            index=:first)                 == - reverse(create_v_test_data(grid, south_panel)[east_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(u, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)            ==         - create_v_test_data(grid, south_panel)[east_indices_first...]
                        @test get_halo_data(getregion(u, panel), North();
                                            operation = :endpoint,
                                            index=:after_last)            ==           create_u_test_data(grid, east_panel)[south_indices_first...]

                        # Confirm that the meridional velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(v, panel), West())  ==           create_v_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(v, panel), South()) ==   reverse(create_u_test_data(grid, south_panel)[east_indices...], dims=2)'
                        @test get_halo_data(getregion(v, panel), North()) ==           create_v_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(v, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)            ==   reverse(create_u_test_data(grid, north_panel)[south_indices_first...])
                        @test get_halo_data(getregion(v, panel), East();
                                            operation=:endpoint,
                                            index=:first)                 == - reverse(create_v_test_data(grid, south_panel)[east_indices_first...])
                        @test get_halo_data(getregion(v, panel), East();
                                            operation=:subset,
                                            index=:first)                 == - reverse(create_u_test_data(grid, east_panel)[south_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                    end
                end
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for face-face-any field pairs such as (Δxᶠᶠᵃ, Δyᶠᶠᵃ)" begin
    for FT in float_types, arch in archs, non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        @info "  Testing fill halos for face-face-any field pairs [$FT, $(typeof(arch)), $cm]..."

        Nx, Ny, Nz = 9, 9, 1

        underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                   panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                   horizontal_direction_halo = 3, non_uniform_conformal_mapping)
        @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        grids = (underlying_grid, immersed_grid)

        for grid in grids
            ψ₁ = Field{Face, Face, Center}(grid)
            ψ₂ = Field{Face, Face, Center}(grid)

            region = Iterate(1:6)
            @apply_regionally ψ₁_data = create_ψ₁_test_data(grid, region)
            @apply_regionally ψ₂_data = create_ψ₂_test_data(grid, region)
            set!(ψ₁, ψ₁_data)
            set!(ψ₂, ψ₂_data)

            fill_halo_regions!((ψ₁, ψ₂))

            Hx, Hy, Hz = halo_size(grid)

            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=nothing, index=:all) # (1:Hx, 1:Ny)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=nothing, index=:all) # (Nx-Hx+1:Nx, 1:Ny)
            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=nothing, index=:all) # (1:Nx, 1:Hy)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=nothing, index=:all) # (1:Nx, Ny-Hy+1:Ny)

            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:first) # (1:Hx, 1)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:first) # (Nx-Hx+1:Nx, 1)
            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:first) # (1, 1:Hy)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:first) # (1, Ny-Hy+1:Ny)

            west_indices_first_shifted_east = first(west_indices_first[1]) + 1 : last(west_indices_first[1]) + 1, west_indices_first[2] # (2:Hx+1, 1)
            east_indices_first_shifted_east = first(east_indices_first[1]) + 1 : last(east_indices_first[1]) + 1, east_indices_first[2] # (Nx-Hx+2:Nx+1, 1)
            north_indices_first_shifted_north = north_indices_first[1], first(north_indices_first[2]) + 1 : last(north_indices_first[2]) + 1 # (1, Ny-Hy+2:Ny+1)
            south_indices_first_shifted_north = south_indices_first[1], first(south_indices_first[2]) + 1 : last(south_indices_first[2]) + 1 # (1, 2:Hy+1)

            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:first) # (1:Hx, 2:Ny)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:first) # (Nx-Hx+1:Nx, 2:Ny)
            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:first) # (2:Nx, 1:Hy)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:first) # (2:Nx, Ny-Hy+1:Ny)

            @allowscalar begin
                for panel in 1:6
                    west_panel = grid.connectivity.connections[panel].west.from_rank
                    east_panel = grid.connectivity.connections[panel].east.from_rank
                    south_panel = grid.connectivity.connections[panel].south.from_rank
                    north_panel = grid.connectivity.connections[panel].north.from_rank

                    if isodd(panel)
                        # Confirm that the ψ₁ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ₁, panel), East())  ==           create_ψ₁_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(ψ₁, panel), South()) ==           create_ψ₁_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ₁, panel), West();
                                            operation=:endpoint,
                                            index=:first)                  ==           create_ψ₂_test_data(grid, south_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), West();
                                            operation=:subset,
                                            index=:first)                  ==   reverse(create_ψ₂_test_data(grid, west_panel)[north_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₁, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₂_test_data(grid, west_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₁_test_data(grid, north_panel)[west_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)             ==   reverse(create_ψ₂_test_data(grid, east_panel)[west_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ₁, panel), North();
                                            operation=:endpoint,
                                            index=:first)                  == - reverse(create_ψ₁_test_data(grid, west_panel)[north_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ₁, panel), North();
                                            operation=:subset,
                                            index=:first)                  == - reverse(create_ψ₂_test_data(grid, north_panel)[west_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₁, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)             ==         - create_ψ₂_test_data(grid, north_panel)[west_indices_first...]

                        # Confirm that the ψ₂ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ₂, panel), East())  ==           create_ψ₂_test_data(grid, east_panel)[west_indices...]
                        @test get_halo_data(getregion(ψ₂, panel), South()) ==           create_ψ₂_test_data(grid, south_panel)[north_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ₂, panel), West();
                                            operation=:endpoint,
                                            index=:first)                  ==         - create_ψ₁_test_data(grid, south_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), West();
                                            operation=:subset,
                                            index=:first)                  == - reverse(create_ψ₁_test_data(grid, west_panel)[north_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₂, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)             ==         - create_ψ₁_test_data(grid, west_panel)[north_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₂_test_data(grid, north_panel)[west_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)             == - reverse(create_ψ₁_test_data(grid, east_panel)[west_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ₂, panel), North();
                                            operation=:endpoint,
                                            index=:first)                  == - reverse(create_ψ₂_test_data(grid, west_panel)[north_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ₂, panel), North();
                                            operation=:subset,
                                            index=:first)                  ==   reverse(create_ψ₁_test_data(grid, north_panel)[west_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₂, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₁_test_data(grid, north_panel)[west_indices_first...]
                    else
                        # Confirm that the ψ₁ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ₁, panel), West())  ==           create_ψ₁_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(ψ₁, panel), North()) ==           create_ψ₁_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ₁, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)             == - reverse(create_ψ₂_test_data(grid, north_panel)[south_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ₁, panel), East();
                                            operation=:endpoint,
                                            index=:first)                  == - reverse(create_ψ₁_test_data(grid, south_panel)[east_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ₁, panel), East();
                                            operation=:subset,
                                            index=:first)                  ==   reverse(create_ψ₂_test_data(grid, east_panel)[south_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₁, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₂_test_data(grid, east_panel)[south_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), South();
                                            operation=:endpoint,
                                            index=:first)                  ==         - create_ψ₂_test_data(grid, west_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), South();
                                            operation=:subset,
                                            index=:first)                  == - reverse(create_ψ₂_test_data(grid, south_panel)[east_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₁, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)             ==         - create_ψ₂_test_data(grid, south_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ₁, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₁_test_data(grid, east_panel)[south_indices_first...]

                        # Confirm that the ψ₂ halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
                        #
                        # Trivial halo checks
                        @test get_halo_data(getregion(ψ₂, panel), West())  ==           create_ψ₂_test_data(grid, west_panel)[east_indices...]
                        @test get_halo_data(getregion(ψ₂, panel), North()) ==           create_ψ₂_test_data(grid, north_panel)[south_indices...]
                        #
                        # Non-trivial halo checks
                        @test get_halo_data(getregion(ψ₂, panel), West();
                                            operation=:endpoint,
                                            index=:after_last)             ==   reverse(create_ψ₁_test_data(grid, north_panel)[south_indices_first_shifted_north...])
                        @test get_halo_data(getregion(ψ₂, panel), East();
                                            operation=:endpoint,
                                            index=:first)                  == - reverse(create_ψ₂_test_data(grid, south_panel)[east_indices_first_shifted_east...])
                        @test get_halo_data(getregion(ψ₂, panel), East();
                                            operation=:subset,
                                            index=:first)                  == - reverse(create_ψ₁_test_data(grid, east_panel)[south_indices_subset_skip_first_index...], dims=1)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₂, panel), East();
                                            operation=:endpoint,
                                            index=:after_last)             ==         - create_ψ₁_test_data(grid, east_panel)[south_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), South();
                                            operation=:endpoint,
                                            index=:first)                  ==           create_ψ₁_test_data(grid, west_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), South();
                                            operation=:subset,
                                            index=:first)                  ==   reverse(create_ψ₁_test_data(grid, south_panel)[east_indices_subset_skip_first_index...], dims=2)'
                        # The index appearing on the LHS above is the index to be skipped.
                        @test get_halo_data(getregion(ψ₂, panel), South();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₁_test_data(grid, south_panel)[east_indices_first...]
                        @test get_halo_data(getregion(ψ₂, panel), North();
                                            operation=:endpoint,
                                            index=:after_last)             ==           create_ψ₂_test_data(grid, east_panel)[south_indices_first...]
                    end
                end
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing simulation on conformal and immersed conformal cubed sphere grids" begin
    for non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        cm_suffix = non_uniform_conformal_mapping ? "NUCM" : "UCM"
        for FT in float_types, arch in archs
            Nx, Ny, Nz = 18, 18, 9

            underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                       panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1,
                                                       horizontal_direction_halo = 6, non_uniform_conformal_mapping)
            @inline bottom(x, y) = ifelse(abs(y) < 30, - 2, 0)
            immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom);
                                                 active_cells_map = true)

            grids = (underlying_grid, immersed_grid)

            for grid in grids
                if grid == underlying_grid
                    @info "  Testing simulation on conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
                    grid_suffix = "UG"
                else
                    @info "  Testing simulation on immersed boundary conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
                    grid_suffix = "IG"
                end

                model = HydrostaticFreeSurfaceModel(grid;
                                                    momentum_advection = WENOVectorInvariant(FT; order=5),
                                                    tracer_advection = WENO(FT; order=5),
                                                    free_surface = SplitExplicitFreeSurface(grid; substeps=12),
                                                    coriolis = HydrostaticSphericalCoriolis(FT),
                                                    tracers = :b,
                                                    buoyancy = BuoyancyTracer())

                simulation = Simulation(model, Δt=1minute, stop_time=10minutes)

                save_fields_interval = 2minute
                checkpointer_interval = 4minutes

                filename_checkpointer =
                    "cubed_sphere_checkpointer_$(FT)_$(typeof(arch))_" * cm_suffix * "_" * grid_suffix
                filename_output_writer = "cubed_sphere_output_$(FT)_$(typeof(arch))_" * cm_suffix * "_" * grid_suffix

                # If previous run produced these files, remove them now to ensure a clean test.
                for f in readdir(".")
                    if f == filename_output_writer * ".jld2" || occursin(r"^" * filename_checkpointer * r"_.*\.jld2$", f)
                        rm(f; force=true)
                    end
                end

                simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                                        schedule = TimeInterval(checkpointer_interval),
                                                                        prefix = filename_checkpointer,
                                                                        overwrite_existing = true)

                outputs = fields(model)
                simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                                schedule = TimeInterval(save_fields_interval),
                                                                filename = filename_output_writer,
                                                                verbose = false,
                                                                overwrite_existing = true)

                run!(simulation)

                @test iteration(simulation) == 10
                @test time(simulation) == 10minutes

                u_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "u"; architecture = CPU())

                if grid == underlying_grid
                    @info "  Restarting simulation from pickup file on conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
                else
                    @info "  Restarting simulation from pickup file on immersed boundary conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
                end

                simulation = Simulation(model, Δt=1minute, stop_time=20minutes)

                simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                                        schedule = TimeInterval(checkpointer_interval),
                                                                        prefix = filename_checkpointer,
                                                                        overwrite_existing = true)

                simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                                schedule = TimeInterval(save_fields_interval),
                                                                filename = filename_output_writer,
                                                                verbose = false,
                                                                overwrite_existing = true)

                run!(simulation, pickup = true)

                @test iteration(simulation) == 20
                @test time(simulation) == 20minutes

                u_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "u"; architecture = CPU())
            end
        end
    end
end

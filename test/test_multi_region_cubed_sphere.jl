include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: halo_size
using Oceananigans.Utils: Iterate, getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!

function get_range_of_indices(operation, index, Nx, Ny)

    if operation == :endpoint && index == :first
        range_x = 1
        range_y = 1
    elseif operation == :endpoint && index == :last
        range_x = Nx
        range_y = Ny
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

function get_halo_data(field, side, k_index=1; operation=nothing, index=:all)

    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)
    
    range_x, range_y = get_range_of_indices(operation, index, Nx, Ny)
    
    if side == :west
        return field.data[-Hx+1:0, range_y, k_index]
    elseif side == :east
        return field.data[Nx+1:Nx+Hx, range_y, k_index]
    elseif side == :south
        return field.data[range_x, -Hy+1:0, k_index]
    elseif side == :north
        return field.data[range_x, Ny+1:Ny+Hy, k_index]
    end
    
end

function get_boundary_indices(Nx, Ny, Hx, Hy, side; operation=nothing, index=:all)
    
    range_x, range_y = get_range_of_indices(operation, index, Nx, Ny)
    
    if side == :west
        return 1:Hx, range_y
    elseif side == :south
        return range_x, 1:Hy
    elseif side == :east
        return Nx-Hx+1:Nx, range_y
    elseif side == :north
        return range_x, Ny-Hy+1:Ny
    end
    
end

"""
    create_test_data(grid, region)

Create an array with integer values of the form, e.g., 541 corresponding to region=5, i=4, j=2.
If `trailing_zeros > 0` then all values are multiplied with `10trailing_zeros`, e.g., for
`trailing_zeros = 2` we have that 54100 corresponds to region=5, i=4, j=2.
"""
function create_test_data(grid, region; trailing_zeros=0)
    Nx, Ny, Nz = size(grid)

    (Nx > 9 || Ny > 9) && error("this won't work; use a grid with Nx, Ny ≤ 9.")

    !(trailing_zeros isa Integer) && error("trailing_zeros has to be an integer")

    factor = trailing_zeros == 0 ? 1 : 10^(trailing_zeros)

    return factor .* [100region + 10i + j for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

create_c_test_data(grid, region) = create_test_data(grid, region, trailing_zeros=0)
create_u_test_data(grid, region) = create_test_data(grid, region, trailing_zeros=1)
create_v_test_data(grid, region) = create_test_data(grid, region, trailing_zeros=2)

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

        Nx, Ny, Nz = size(grid_cs32)
        radius = getregion(grid_cs32, 1).radius

        # construct a ConformalCubedSphereGrid similar to cs32
        grid = ConformalCubedSphereGrid(arch; z, panel_size=(Nx, Ny, Nz), radius)

        for panel in 1:6

            CUDA.@allowscalar begin
                # Test only on cca and ffa; fca and cfa are all zeros on grid_cs32!
                # Only test interior points since halo regions are not filled for grid_cs32!

                @test isapprox(getregion(grid, panel).φᶜᶜᵃ[1:Nx, 1:Ny], getregion(grid_cs32, panel).φᶜᶜᵃ[1:Nx, 1:Ny])
                @test isapprox(getregion(grid, panel).λᶜᶜᵃ[1:Nx, 1:Ny], getregion(grid_cs32, panel).λᶜᶜᵃ[1:Nx, 1:Ny])

                # before we test, make sure we don't consider +180 and -180 longitudes as being "different"
                getregion(grid, panel).λᶠᶠᵃ[getregion(grid, panel).λᶠᶠᵃ .≈ -180] .= 180

                # and if poles are included, they have the same longitude.
                getregion(grid, panel).λᶠᶠᵃ[getregion(grid, panel).φᶠᶠᵃ .≈ +90] = getregion(grid_cs32, panel).λᶠᶠᵃ[getregion(grid, panel).φᶠᶠᵃ .≈ +90]
                getregion(grid, panel).λᶠᶠᵃ[getregion(grid, panel).φᶠᶠᵃ .≈ -90] = getregion(grid_cs32, panel).λᶠᶠᵃ[getregion(grid, panel).φᶠᶠᵃ .≈ -90]
                @test isapprox(getregion(grid, panel).φᶠᶠᵃ[1:Nx, 1:Ny], getregion(grid_cs32, panel).φᶠᶠᵃ[1:Nx, 1:Ny])
                @test isapprox(getregion(grid, panel).λᶠᶠᵃ[1:Nx, 1:Ny], getregion(grid_cs32, panel).λᶠᶠᵃ[1:Nx, 1:Ny])
            end
        end
    end
end

panel_sizes = ((8, 8, 1), (9, 9, 2))

@testset "Testing area metrics" begin
    for FT in float_types
        for arch in archs
            for panel_size in panel_sizes
                Nx, Ny, Nz = panel_size

                grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1)

                areaᶜᶜᵃ = areaᶠᶜᵃ = areaᶜᶠᵃ = areaᶠᶠᵃ = 0

                for region in 1:length(grid.partition)

                    region_Nx, region_Ny, _ = size(getregion(grid, region))

                    areaᶜᶜᵃ += sum(getregion(grid, region).Azᶜᶜᵃ[1:region_Nx, 1:region_Ny])
                    areaᶠᶜᵃ += sum(getregion(grid, region).Azᶠᶜᵃ[1:region_Nx, 1:region_Ny])
                    areaᶜᶠᵃ += sum(getregion(grid, region).Azᶜᶠᵃ[1:region_Nx, 1:region_Ny])
                    areaᶠᶠᵃ += sum(getregion(grid, region).Azᶠᶠᵃ[1:region_Nx, 1:region_Ny])
                end

                @test areaᶜᶜᵃ ≈ areaᶠᶜᵃ ≈ areaᶜᶠᵃ ≈ areaᶠᶠᵃ ≈ 4π * grid.radius^2
            end
        end
    end
end


@testset "Testing conformal cubed sphere metric/coordinate halo filling" begin
    for FT in float_types
        for arch in archs
            Nx, Ny, Nz = 3, 3, 1

            grid         = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1)
            grid_bounded = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_topology = Bounded)

            @info "  Testing conformal cubed sphere face-coordinate halos [$FT, $(typeof(arch))]..."

            for region in 1:6
                @test getregion(grid, region).φᶠᶜᵃ[1:Nx+1, 1:Ny]  ≈ getregion(grid_bounded, region).φᶠᶜᵃ[1:Nx+1, 1:Ny]
                @test getregion(grid, region).λᶠᶜᵃ[1:Nx+1, 1:Ny]  ≈ getregion(grid_bounded, region).λᶠᶜᵃ[1:Nx+1, 1:Ny]

                @test getregion(grid, region).φᶜᶠᵃ[1:Nx, 1:Ny+1]  ≈ getregion(grid_bounded, region).φᶜᶠᵃ[1:Nx, 1:Ny+1]
                @test getregion(grid, region).λᶜᶠᵃ[1:Nx, 1:Ny+1]  ≈ getregion(grid_bounded, region).λᶜᶠᵃ[1:Nx, 1:Ny+1]

                @test getregion(grid, region).φᶠᶠᵃ[1:Nx+1, 1:Ny+1]  ≈ getregion(grid_bounded, region).φᶠᶠᵃ[1:Nx+1, 1:Ny+1]
                @test getregion(grid, region).λᶠᶠᵃ[1:Nx+1, 1:Ny+1]  ≈ getregion(grid_bounded, region).λᶠᶠᵃ[1:Nx+1, 1:Ny+1]
            end

            @info "  Testing conformal cubed sphere face-metric halos [$FT, $(typeof(arch))]..."

            for region in 1:6
                @test getregion(grid, region).Δxᶠᶜᵃ[1:Nx+1, 1:Ny] ≈ getregion(grid_bounded, region).Δxᶠᶜᵃ[1:Nx+1, 1:Ny]
                @test getregion(grid, region).Δyᶠᶜᵃ[1:Nx+1, 1:Ny] ≈ getregion(grid_bounded, region).Δyᶠᶜᵃ[1:Nx+1, 1:Ny]
                @test getregion(grid, region).Azᶠᶜᵃ[1:Nx+1, 1:Ny] ≈ getregion(grid_bounded, region).Azᶠᶜᵃ[1:Nx+1, 1:Ny]

                @test getregion(grid, region).Δxᶜᶠᵃ[1:Nx, 1:Ny+1] ≈ getregion(grid_bounded, region).Δxᶜᶠᵃ[1:Nx, 1:Ny+1]
                @test getregion(grid, region).Δyᶜᶠᵃ[1:Nx, 1:Ny+1] ≈ getregion(grid_bounded, region).Δyᶜᶠᵃ[1:Nx, 1:Ny+1]
                @test getregion(grid, region).Azᶜᶠᵃ[1:Nx, 1:Ny+1] ≈ getregion(grid_bounded, region).Azᶜᶠᵃ[1:Nx, 1:Ny+1]

                @test getregion(grid, region).Δxᶠᶠᵃ[1:Nx+1, 1:Ny+1] ≈ getregion(grid_bounded, region).Δxᶠᶠᵃ[1:Nx+1, 1:Ny+1]
                @test getregion(grid, region).Δyᶠᶠᵃ[1:Nx+1, 1:Ny+1] ≈ getregion(grid_bounded, region).Δyᶠᶠᵃ[1:Nx+1, 1:Ny+1]
                @test getregion(grid, region).Azᶠᶠᵃ[1:Nx+1, 1:Ny+1] ≈ getregion(grid_bounded, region).Azᶠᶠᵃ[1:Nx+1, 1:Ny+1]
            end
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for tracers" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos for tracers [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo=2)
            c = CenterField(grid)

            region = Iterate(1:6)
            @apply_regionally data = create_c_test_data(grid, region)
            set!(c, data)
            fill_halo_regions!(c)

            Hx, Hy, Hz = halo_size(c.grid)

            west_indices  = 1:Hx, 1:Ny
            south_indices = 1:Nx, 1:Hy
            east_indices  = Nx-Hx+1:Nx, 1:Ny
            north_indices = 1:Nx, Ny-Hy+1:Ny

            # Confirm that the tracer halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
                switch_device!(grid, 1)
                @test get_halo_data(getregion(c, 1), :west)  == reverse(create_c_test_data(grid, 5)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 1), :east)  ==         create_c_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(c, 1), :south) ==         create_c_test_data(grid, 6)[north_indices...]
                @test get_halo_data(getregion(c, 1), :north) == reverse(create_c_test_data(grid, 3)[west_indices...], dims=2)'

                switch_device!(grid, 2)
                @test get_halo_data(getregion(c, 2), :west)  ==         create_c_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(c, 2), :east)  == reverse(create_c_test_data(grid, 4)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 2), :south) == reverse(create_c_test_data(grid, 6)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 2), :north) ==         create_c_test_data(grid, 3)[south_indices...]

                switch_device!(grid, 3)
                @test get_halo_data(getregion(c, 3), :west)  == reverse(create_c_test_data(grid, 1)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 3), :east)  ==         create_c_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(c, 3), :south) ==         create_c_test_data(grid, 2)[north_indices...]
                @test get_halo_data(getregion(c, 3), :north) == reverse(create_c_test_data(grid, 5)[west_indices...], dims=2)'

                switch_device!(grid, 4)
                @test get_halo_data(getregion(c, 4), :west)  ==         create_c_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(c, 4), :east)  == reverse(create_c_test_data(grid, 6)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 4), :south) == reverse(create_c_test_data(grid, 2)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 4), :north) ==         create_c_test_data(grid, 5)[south_indices...]

                switch_device!(grid, 5)
                @test get_halo_data(getregion(c, 5), :west)  == reverse(create_c_test_data(grid, 3)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 5), :east)  ==         create_c_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(c, 5), :south) ==         create_c_test_data(grid, 4)[north_indices...]
                @test get_halo_data(getregion(c, 5), :north) == reverse(create_c_test_data(grid, 1)[west_indices...], dims=2)'

                switch_device!(grid, 6)
                @test get_halo_data(getregion(c, 6), :west)  ==         create_c_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(c, 6), :east)  == reverse(create_c_test_data(grid, 2)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 6), :south) == reverse(create_c_test_data(grid, 4)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 6), :north) ==         create_c_test_data(grid, 1)[south_indices...]
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for horizontal velocities" begin
    for FT in float_types
        for arch in archs

            @info "  Testing fill halos for horizontal velocities [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3)

            u = XFaceField(grid)
            v = YFaceField(grid)
            
            region = Iterate(1:6)
            @apply_regionally u_data = create_u_test_data(grid, region)
            @apply_regionally v_data = create_v_test_data(grid, region)
            set!(u, u_data)
            set!(v, v_data)

            # we need 2 halo filling passes for velocities at the moment
            for _ in 1:2
                fill_halo_regions!(u)
                fill_halo_regions!(v)
                @apply_regionally replace_horizontal_velocity_halos!((; u, v, w = nothing), grid)
            end

            Hx, Hy, Hz = halo_size(u.grid)

            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, :south; operation=nothing, index=:all)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, :east;  operation=nothing, index=:all)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, :north; operation=nothing, index=:all)
            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, :west;  operation=nothing, index=:all)

            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, :south; operation=:endpoint, index=:first)
            south_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, :south; operation=:endpoint, index=:last)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, :east;  operation=:endpoint, index=:first)
            east_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, :east;  operation=:endpoint, index=:last)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, :north; operation=:endpoint, index=:first)
            north_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, :north; operation=:endpoint, index=:last)
            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, :west;  operation=:endpoint, index=:first)
            west_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, :west;  operation=:endpoint, index=:last)

            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, :south; operation=:subset, index=:first)
            south_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, :south; operation=:subset, index=:last)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, :east;  operation=:subset, index=:first)
            east_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, :east;  operation=:subset, index=:last)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, :north; operation=:subset, index=:first)
            north_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, :north; operation=:subset, index=:last)
            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, :west;  operation=:subset, index=:first)
            west_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, :west;  operation=:subset, index=:last)

            # Confirm that the zonal velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin

                # Panel 1
                switch_device!(grid, 1)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 1), :west)  ==   reverse(create_v_test_data(grid, 5)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 1), :east)  ==           create_u_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(u, 1), :south) ==           create_u_test_data(grid, 6)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 1), :north;
                                    operation=:subset, 
                                    index=:first)            == - reverse(create_v_test_data(grid, 3)[west_indices_subset_skip_first_index...], dims=2)'        
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 1), :north;
                                    operation=:endpoint, 
                                    index=:first)            == - reverse(create_u_test_data(grid, 5)[north_indices_first...])

                # Panel 2
                switch_device!(grid, 2)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 2), :west)  ==           create_u_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(u, 2), :east)  ==   reverse(create_v_test_data(grid, 4)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 2), :north) ==           create_u_test_data(grid, 3)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 2), :south;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_v_test_data(grid, 6)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 2), :south;
                                    operation=:endpoint,
                                    index=:first)            ==         - create_v_test_data(grid, 1)[east_indices_first...]

                # Panel 3
                switch_device!(grid, 3)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 3), :west)  ==   reverse(create_v_test_data(grid, 1)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 3), :east)  ==           create_u_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(u, 3), :south) ==           create_u_test_data(grid, 2)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 3), :north;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_v_test_data(grid, 5)[west_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 3), :north;
                                    operation=:endpoint,
                                    index=:first)            ==         - reverse(create_u_test_data(grid, 1)[north_indices_first...])

                # Panel 4
                switch_device!(grid, 4)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 4), :west)  ==           create_u_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(u, 4), :east)  ==   reverse(create_v_test_data(grid, 6)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 4), :north) ==           create_u_test_data(grid, 5)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 4), :south;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_v_test_data(grid, 2)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 4), :south;
                                    operation=:endpoint, 
                                    index=:first)            ==         - create_v_test_data(grid, 3)[east_indices_first...]

                # Panel 5
                switch_device!(grid, 5)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 5), :west)  ==   reverse(create_v_test_data(grid, 3)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 5), :east)  ==           create_u_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(u, 5), :south) ==           create_u_test_data(grid, 4)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 5), :north;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_v_test_data(grid, 1)[west_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 5), :north;
                                    operation=:endpoint,
                                    index=:first)            ==  - reverse(create_u_test_data(grid, 3)[north_indices_first...])

                # Panel 6
                switch_device!(grid, 6)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 6), :west)  ==           create_u_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(u, 6), :east)  ==   reverse(create_v_test_data(grid, 2)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 6), :north) ==           create_u_test_data(grid, 1)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 6), :south;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_v_test_data(grid, 4)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 6), :south;
                                    operation=:endpoint,
                                    index=:first)            ==         - create_v_test_data(grid, 5)[east_indices_first...]
            end # CUDA.@allowscalar

            # Confirm that the meridional velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin

                # Panel 1
                switch_device!(grid, 1)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 1), :east)  ==           create_v_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(v, 1), :south) ==           create_v_test_data(grid, 6)[north_indices...]
                @test get_halo_data(getregion(v, 1), :north) ==   reverse(create_u_test_data(grid, 3)[west_indices...], dims=2)'

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 1), :west;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 5)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 1), :west;
                                    operation=:endpoint,
                                    index=:first)            ==         - create_u_test_data(grid, 6)[north_indices_first...]

                # Panel 2
                switch_device!(grid, 2)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 2), :west)  ==           create_v_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(v, 2), :south) ==   reverse(create_u_test_data(grid, 6)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 2), :north) ==           create_v_test_data(grid, 3)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 2), :east;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 4)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 2), :east; 
                                    operation=:endpoint,
                                    index=:first)            == - reverse(create_v_test_data(grid, 6)[east_indices_first...])

                # Panel 3
                switch_device!(grid, 3)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 3), :east)  ==           create_v_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(v, 3), :south) ==           create_v_test_data(grid, 2)[north_indices...]
                @test get_halo_data(getregion(v, 3), :north) ==   reverse(create_u_test_data(grid, 5)[west_indices...], dims=2)'           

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 3), :west;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 1)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 3), :west; 
                                    operation=:endpoint,
                                    index=:first)            ==         - create_u_test_data(grid, 2)[north_indices_first...]

                # Panel 4
                switch_device!(grid, 4)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 4), :west)  ==           create_v_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(v, 4), :south) ==   reverse(create_u_test_data(grid, 2)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 4), :north) ==           create_v_test_data(grid, 5)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 4), :east;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 6)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 4), :east; 
                                    operation=:endpoint,
                                    index=:first)            == - reverse(create_v_test_data(grid, 2)[east_indices_first...])

                # Panel 5
                switch_device!(grid, 5)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 5), :east)  ==           create_v_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(v, 5), :south) ==           create_v_test_data(grid, 4)[north_indices...]
                @test get_halo_data(getregion(v, 5), :north) ==   reverse(create_u_test_data(grid, 1)[west_indices...], dims=2)'

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 5), :west;
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 3)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 5), :west; 
                                    operation=:endpoint,
                                    index=:first)            ==         - create_u_test_data(grid, 4)[north_indices_first...] 

                # Panel 6
                switch_device!(grid, 6)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 6), :west)  ==           create_v_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(v, 6), :south) ==   reverse(create_u_test_data(grid, 4)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 6), :north) ==           create_v_test_data(grid, 1)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 6), :east;
                                    operation=:subset,
                                    index=:first) == - reverse(create_u_test_data(grid, 2)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 6), :east;
                                    operation=:endpoint,
                                    index=:first)            == - reverse(create_v_test_data(grid, 4)[east_indices_first...])
            end # CUDA.@allowscalar
        end
    end
end

include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.Utils: Iterate, getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!

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

# Solid body rotation
R = 1        # sphere's radius
U = 1        # velocity scale
φʳ = 0       # Latitude pierced by the axis of rotation
α  = 90 - φʳ # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

"""
    create_test_data(grid, region)

Create an array with integer values of the form, e.g., 541 corresponding to region=5, i=4, j=2.
If `trailing_zeros > 0` then all values are multiplied with `10^trailing_zeros`, e.g., for
`trailing_zeros = 2` we have that 54100 corresponds to region=5, i=4, j=2.
"""
function create_test_data(grid, region; trailing_zeros=0)
    Nx, Ny, Nz = size(grid)
    (Nx > 9 || Ny > 9) && error("you provided (Nx, Ny) = ($Nx, $Ny); use a grid with Nx, Ny ≤ 9.")
    !(trailing_zeros isa Integer) && error("trailing_zeros has to be an integer")
    factor = 10^(trailing_zeros)

    return factor .* [100region + 10i + j for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

create_c_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=0)
create_ψ_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=1)

create_u_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=2)
create_v_test_data(grid, region) = create_test_data(grid, region; trailing_zeros=3)

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
        grid = conformal_cubed_sphere_panel(cs32_filepath; panel, Nz, z)
        @test grid isa OrthogonalSphericalShellGrid
    end

    for arch in archs
        @info "  Testing conformal cubed sphere face grid from file [$(typeof(arch))]..."

        # read cs32 grid from file
        grid_cs32 = ConformalCubedSphereGrid(cs32_filepath, arch; Nz, z)

        radius = first(grid_cs32).radius
        Nx, Ny, Nz = size(grid_cs32)
        Hx, Hy, Hz = halo_size(grid_cs32)
        Hx !== Hy && error("Hx must be same as Hy")

        # construct a ConformalCubedSphereGrid similar to cs32
        grid = ConformalCubedSphereGrid(arch; z, panel_size=(Nx, Ny, Nz), radius,
                                        horizontal_direction_halo = Hx, z_halo = Hz)

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

                    CUDA.@allowscalar begin
                        areaᶜᶜᵃ += sum(getregion(grid, region).Azᶜᶜᵃ[1:region_Nx, 1:region_Ny])
                        areaᶠᶜᵃ += sum(getregion(grid, region).Azᶠᶜᵃ[1:region_Nx, 1:region_Ny])
                        areaᶜᶠᵃ += sum(getregion(grid, region).Azᶜᶠᵃ[1:region_Nx, 1:region_Ny])
                        areaᶠᶠᵃ += sum(getregion(grid, region).Azᶠᶠᵃ[1:region_Nx, 1:region_Ny])
                    end
                end

                @test areaᶜᶜᵃ ≈ areaᶠᶜᵃ ≈ areaᶜᶠᵃ ≈ areaᶠᶠᵃ ≈ 4π * grid.radius^2
            end
        end
    end
end

#=
# TODO
@testset "Testing conformal cubed sphere metric/coordinate halo filling" begin
    for FT in float_types
        for arch in archs
            Nx, Ny, Nz = 3, 3, 1

            grid         = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1)

            @info "  Testing conformal cubed sphere face-coordinate halos [$FT, $(typeof(arch))]..."

            # ...

            @info "  Testing conformal cubed sphere face-metric halos [$FT, $(typeof(arch))]..."

            # ...
        end
    end
end
=#

@testset "Testing conformal cubed sphere fill halos for tracers" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos for tracers [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3)
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
                @test get_halo_data(getregion(c, 1), West())  == reverse(create_c_test_data(grid, 5)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 1), East())  ==         create_c_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(c, 1), South()) ==         create_c_test_data(grid, 6)[north_indices...]
                @test get_halo_data(getregion(c, 1), North()) == reverse(create_c_test_data(grid, 3)[west_indices...], dims=2)'

                switch_device!(grid, 2)
                @test get_halo_data(getregion(c, 2), West())  ==         create_c_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(c, 2), East())  == reverse(create_c_test_data(grid, 4)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 2), South()) == reverse(create_c_test_data(grid, 6)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 2), North()) ==         create_c_test_data(grid, 3)[south_indices...]

                switch_device!(grid, 3)
                @test get_halo_data(getregion(c, 3), West())  == reverse(create_c_test_data(grid, 1)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 3), East())  ==         create_c_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(c, 3), South()) ==         create_c_test_data(grid, 2)[north_indices...]
                @test get_halo_data(getregion(c, 3), North()) == reverse(create_c_test_data(grid, 5)[west_indices...], dims=2)'

                switch_device!(grid, 4)
                @test get_halo_data(getregion(c, 4), West())  ==         create_c_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(c, 4), East())  == reverse(create_c_test_data(grid, 6)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 4), South()) == reverse(create_c_test_data(grid, 2)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 4), North()) ==         create_c_test_data(grid, 5)[south_indices...]

                switch_device!(grid, 5)
                @test get_halo_data(getregion(c, 5), West())  == reverse(create_c_test_data(grid, 3)[north_indices...], dims=1)'
                @test get_halo_data(getregion(c, 5), East())  ==         create_c_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(c, 5), South()) ==         create_c_test_data(grid, 4)[north_indices...]
                @test get_halo_data(getregion(c, 5), North()) == reverse(create_c_test_data(grid, 1)[west_indices...], dims=2)'

                switch_device!(grid, 6)
                @test get_halo_data(getregion(c, 6), West())  ==         create_c_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(c, 6), East())  == reverse(create_c_test_data(grid, 2)[south_indices...], dims=1)'
                @test get_halo_data(getregion(c, 6), South()) == reverse(create_c_test_data(grid, 4)[east_indices...], dims=2)'
                @test get_halo_data(getregion(c, 6), North()) ==         create_c_test_data(grid, 1)[south_indices...]
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for horizontal velocities" begin
    for FT in float_types
        for arch in archs

            @info "  Testing fill halos for horizontal velocities [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 3, 3, 1

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
                @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
            end

            Hx, Hy, Hz = halo_size(u.grid)

            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=nothing, index=:all)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=nothing, index=:all)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=nothing, index=:all)
            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=nothing, index=:all)

            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:first)
            south_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:last)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:first)
            east_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:last)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:first)
            north_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:last)
            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:first)
            west_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:last)

            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:first)
            south_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:last)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:first)
            east_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:last)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:first)
            north_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:last)
            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:first)
            west_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:last)

            # Confirm that the zonal velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin

                # Panel 1
                switch_device!(grid, 1)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 1), West())  ==   reverse(create_v_test_data(grid, 5)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 1), East())  ==           create_u_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(u, 1), South()) ==           create_u_test_data(grid, 6)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 1), North();
                                    operation=:subset, 
                                    index=:first) == - reverse(create_v_test_data(grid, 3)[west_indices_subset_skip_first_index...], dims=2)'        
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 1), North();
                                    operation=:endpoint, 
                                    index=:first) == - reverse(create_u_test_data(grid, 5)[north_indices_first...])

                
                # Panel 2
                switch_device!(grid, 2)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 2), West())  ==           create_u_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(u, 2), East())  ==   reverse(create_v_test_data(grid, 4)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 2), North()) ==           create_u_test_data(grid, 3)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 2), South();
                                    operation=:subset,
                                    index=:first) == - reverse(create_v_test_data(grid, 6)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 2), South();
                                    operation=:endpoint,
                                    index=:first) == - create_v_test_data(grid, 1)[east_indices_first...]

                # Panel 3
                switch_device!(grid, 3)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 3), West())  ==   reverse(create_v_test_data(grid, 1)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 3), East())  ==           create_u_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(u, 3), South()) ==           create_u_test_data(grid, 2)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 3), North();
                                    operation=:subset,
                                    index=:first) == - reverse(create_v_test_data(grid, 5)[west_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 3), North();
                                    operation=:endpoint,
                                    index=:first) == - reverse(create_u_test_data(grid, 1)[north_indices_first...])

                # Panel 4
                switch_device!(grid, 4)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 4), West())  ==           create_u_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(u, 4), East())  ==   reverse(create_v_test_data(grid, 6)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 4), North()) ==           create_u_test_data(grid, 5)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 4), South();
                                    operation=:subset,
                                    index=:first) == - reverse(create_v_test_data(grid, 2)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 4), South();
                                    operation=:endpoint, 
                                    index=:first) == - create_v_test_data(grid, 3)[east_indices_first...]

                # Panel 5
                switch_device!(grid, 5)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 5), West())  ==   reverse(create_v_test_data(grid, 3)[north_indices...], dims=1)'
                @test get_halo_data(getregion(u, 5), East())  ==           create_u_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(u, 5), South()) ==           create_u_test_data(grid, 4)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 5), North();
                                    operation=:subset,
                                    index=:first) == - reverse(create_v_test_data(grid, 1)[west_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 5), North();
                                    operation=:endpoint,
                                    index=:first) ==  - reverse(create_u_test_data(grid, 3)[north_indices_first...])

                # Panel 6
                switch_device!(grid, 6)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(u, 6), West())  ==           create_u_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(u, 6), East())  ==   reverse(create_v_test_data(grid, 2)[south_indices...], dims=1)'
                @test get_halo_data(getregion(u, 6), North()) ==           create_u_test_data(grid, 1)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(u, 6), South();
                                    operation=:subset,
                                    index=:first) == - reverse(create_v_test_data(grid, 4)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(u, 6), South();
                                    operation=:endpoint,
                                    index=:first) == - create_v_test_data(grid, 5)[east_indices_first...]
            end # CUDA.@allowscalar

            # Confirm that the meridional velocity halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
                # Panel 1
                switch_device!(grid, 1)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 1), East())  ==           create_v_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(v, 1), South()) ==           create_v_test_data(grid, 6)[north_indices...]
                @test get_halo_data(getregion(v, 1), North()) ==   reverse(create_u_test_data(grid, 3)[west_indices...], dims=2)'

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 1), West();
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 5)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 1), West();
                                    operation=:endpoint,
                                    index=:first)            ==         - create_u_test_data(grid, 6)[north_indices_first...]
                
                # Panel 2
                switch_device!(grid, 2)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 2), West())  ==           create_v_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(v, 2), South()) ==   reverse(create_u_test_data(grid, 6)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 2), North()) ==           create_v_test_data(grid, 3)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 2), East();
                                    operation=:subset,
                                    index=:first) == - reverse(create_u_test_data(grid, 4)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 2), East(); 
                                    operation=:endpoint,
                                    index=:first) == - reverse(create_v_test_data(grid, 6)[east_indices_first...])

                # Panel 3
                switch_device!(grid, 3)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 3), East())  ==           create_v_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(v, 3), South()) ==           create_v_test_data(grid, 2)[north_indices...]
                @test get_halo_data(getregion(v, 3), North()) ==   reverse(create_u_test_data(grid, 5)[west_indices...], dims=2)'           

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 3), West();
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 1)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 3), West(); 
                                    operation=:endpoint,
                                    index=:first) == - create_u_test_data(grid, 2)[north_indices_first...]

                # Panel 4
                switch_device!(grid, 4)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 4), West())  ==           create_v_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(v, 4), South()) ==   reverse(create_u_test_data(grid, 2)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 4), North()) ==           create_v_test_data(grid, 5)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 4), East();
                                    operation=:subset,
                                    index=:first) == - reverse(create_u_test_data(grid, 6)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 4), East(); 
                                    operation=:endpoint,
                                    index=:first) == - reverse(create_v_test_data(grid, 2)[east_indices_first...])

                # Panel 5
                switch_device!(grid, 5)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 5), East())  ==           create_v_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(v, 5), South()) ==           create_v_test_data(grid, 4)[north_indices...]
                @test get_halo_data(getregion(v, 5), North()) ==   reverse(create_u_test_data(grid, 1)[west_indices...], dims=2)'

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 5), West();
                                    operation=:subset,
                                    index=:first)            == - reverse(create_u_test_data(grid, 3)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 5), West(); 
                                    operation=:endpoint,
                                    index=:first) == - create_u_test_data(grid, 4)[north_indices_first...] 

                # Panel 6
                switch_device!(grid, 6)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(v, 6), West())  ==           create_v_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(v, 6), South()) ==   reverse(create_u_test_data(grid, 4)[east_indices...], dims=2)'
                @test get_halo_data(getregion(v, 6), North()) ==           create_v_test_data(grid, 1)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(v, 6), East();
                                    operation=:subset,
                                    index=:first) == - reverse(create_u_test_data(grid, 2)[south_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(v, 6), East();
                                    operation=:endpoint,
                                    index=:first) == - reverse(create_v_test_data(grid, 4)[east_indices_first...])
            end # CUDA.@allowscalar
        end
    end
end

@testset "Testing conformal cubed sphere fill halos for Face-Face-Any field" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos for streamfunction [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3)
            ψ = Field{Face, Face, Center}(grid)

            region = Iterate(1:6)
            @apply_regionally data = create_ψ_test_data(grid, region)
            set!(ψ, data)

            for _ in 1:2
                fill_halo_regions!(ψ)
            end

            Hx, Hy, Hz = halo_size(ψ.grid)

            south_indices = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=nothing, index=:all)
            east_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=nothing, index=:all)
            north_indices = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=nothing, index=:all)
            west_indices  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=nothing, index=:all)

            south_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:first)
            south_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:endpoint, index=:last)
            east_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:first)
            east_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:endpoint, index=:last)
            north_indices_first = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:first)
            north_indices_last  = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:endpoint, index=:last)
            west_indices_first  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:first)
            west_indices_last   = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:endpoint, index=:last)

            south_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:first)
            south_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, South(); operation=:subset, index=:last)
            east_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:first)
            east_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, East();  operation=:subset, index=:last)
            north_indices_subset_skip_first_index = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:first)
            north_indices_subset_skip_last_index  = get_boundary_indices(Nx, Ny, Hx, Hy, North(); operation=:subset, index=:last)
            west_indices_subset_skip_first_index  = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:first)
            west_indices_subset_skip_last_index   = get_boundary_indices(Nx, Ny, Hx, Hy, West();  operation=:subset, index=:last)

            # Confirm that the tracer halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin

                # Panel 1
                switch_device!(grid, 1)

                # Trivial halo checks with no off-set in index
                @test get_halo_data(getregion(ψ, 1), East())  ==         create_ψ_test_data(grid, 2)[west_indices...]
                @test get_halo_data(getregion(ψ, 1), South()) ==         create_ψ_test_data(grid, 6)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 1), North();
                                    operation=:subset, 
                                    index=:first)             == reverse(create_ψ_test_data(grid, 3)[west_indices_subset_skip_first_index...], dims=2)'        
                # Currently we do not have any test for the point of intersection of the northwest (halo) corners of panels 1, 3, and 5.

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 1), West();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 5)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 1), West();
                                    operation=:endpoint,
                                    index=:first)             ==         create_ψ_test_data(grid, 6)[north_indices_first...]

                switch_device!(grid, 2)
                @test get_halo_data(getregion(ψ, 2), West())  ==         create_ψ_test_data(grid, 1)[east_indices...]
                @test get_halo_data(getregion(ψ, 2), North()) ==         create_ψ_test_data(grid, 3)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 2), East();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 4)[south_indices_subset_skip_first_index...], dims=1)'
                # Currently we do not have any test for the point of intersection of the southeast (halo) corners of panels 2, 4, and 6.

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 2), South();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 6)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 2), South();
                                    operation=:endpoint,
                                    index=:first)             ==         create_ψ_test_data(grid, 1)[east_indices_first...]                

                switch_device!(grid, 3)
                @test get_halo_data(getregion(ψ, 3), East())  ==         create_ψ_test_data(grid, 4)[west_indices...]
                @test get_halo_data(getregion(ψ, 3), South()) ==         create_ψ_test_data(grid, 2)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 3), West();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 1)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 3), West(); 
                                    operation=:endpoint,
                                    index=:first)             ==         create_ψ_test_data(grid, 2)[north_indices_first...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 3), North();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 5)[west_indices_subset_skip_first_index...], dims=2)'
                # Currently we do not have any test for the point of intersection of the northwest (halo) corners of panels 1, 3, and 5.

                switch_device!(grid, 4)
                @test get_halo_data(getregion(ψ, 4), West())  ==         create_ψ_test_data(grid, 3)[east_indices...]
                @test get_halo_data(getregion(ψ, 4), North()) ==         create_ψ_test_data(grid, 5)[south_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 4), East();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 6)[south_indices_subset_skip_first_index...], dims=1)'
                # Currently we do not have any test for the point of intersection of the southeast (halo) corners of panels 2, 4, and 6.

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 4), South();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 2)[east_indices_subset_skip_first_index...], dims=2)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 4), South();
                                    operation=:endpoint, 
                                    index=:first)             ==         create_ψ_test_data(grid, 3)[east_indices_first...]

                switch_device!(grid, 5)
                @test get_halo_data(getregion(ψ, 5), East())  ==         create_ψ_test_data(grid, 6)[west_indices...]
                @test get_halo_data(getregion(ψ, 5), South()) ==         create_ψ_test_data(grid, 4)[north_indices...]

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 5), West();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 3)[north_indices_subset_skip_first_index...], dims=1)'
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 5), West(); 
                                    operation=:endpoint,
                                    index=:first)             ==         create_ψ_test_data(grid, 4)[north_indices_first...] 

                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 5), North();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 1)[west_indices_subset_skip_first_index...], dims=2)'
                # Currently we do not have any test for the point of intersection of the northwest (halo) corners of panels 1, 3, and 5.

                switch_device!(grid, 6)
                @test get_halo_data(getregion(ψ, 6), West())  ==         create_ψ_test_data(grid, 5)[east_indices...]
                @test get_halo_data(getregion(ψ, 6), North()) ==         create_ψ_test_data(grid, 1)[south_indices...]
                
                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 6), East();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 2)[south_indices_subset_skip_first_index...], dims=1)'
                # Currently we do not have any test for the point of intersection of the southeast (halo) corners of panels 2, 4, and 6.
                
                # Non-trivial halo checks with off-set in index
                @test get_halo_data(getregion(ψ, 6), South();
                                    operation=:subset,
                                    index=:first)             == reverse(create_ψ_test_data(grid, 4)[east_indices_subset_skip_first_index...], dims=2)'
                
                # The index appearing on the LHS above is the index to be skipped.
                @test get_halo_data(getregion(ψ, 6), South();
                                    operation=:endpoint,
                                    index=:first)             == create_ψ_test_data(grid, 5)[east_indices_first...]

            end # CUDA.@allowscalar
        end
    end
end 

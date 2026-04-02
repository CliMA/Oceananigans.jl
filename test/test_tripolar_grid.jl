include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.Grids: get_cartesian_nodes_and_vertices, RightFaceFolded, RightCenterFolded
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.BoundaryConditions: Zipper, FPivot, UPivot

using Oceananigans.Utils: KernelParameters
import Oceananigans.Utils: contiguousrange

contiguousrange(::KernelParameters{spec, offset}) where {spec, offset} = contiguousrange(spec, offset)
fold_topologies = (RightCenterFolded, RightFaceFolded)

@kernel function compute_nonorthogonality_angle!(angle, grid, xF, yF, zF)
    i, j = @index(Global, NTuple)

    @inbounds begin
        x⁻ = xF[i, j]
        y⁻ = yF[i, j]
        z⁻ = zF[i, j]

        x⁺¹ = xF[i + 1, j]
        y⁺¹ = yF[i + 1, j]
        z⁺¹ = zF[i + 1, j]
        x⁺² = xF[i, j + 1]
        y⁺² = yF[i, j + 1]
        z⁺² = zF[i, j + 1]

        v1 = (x⁺¹ - x⁻, y⁺¹ - y⁻, z⁺¹ - z⁻)
        v2 = (x⁺² - x⁻, y⁺² - y⁻, z⁺² - z⁻)

        # Check orthogonality by computing the angle between the vectors
        cosθ = dot(v1, v2) / (norm(v1) * norm(v2))
        immersed = immersed_cell(i, j, 1, grid)
        angle[i, j] = ifelse(immersed, π / 2, acos(cosθ)) - π / 2

        # convert to degrees
        angle[i, j] = rad2deg(angle[i, j])
    end
end


@testset "Unit tests..." begin
    for arch in archs
        @testset "$fold_topology fold topology" for fold_topology in fold_topologies
            first_pole_longitude = 75
            north_poles_latitude = 35
            southernmost_latitude = -35
            grid = TripolarGrid(arch;
                                size = (4, 5, 1),
                                z = (0, 1),
                                first_pole_longitude,
                                north_poles_latitude,
                                southernmost_latitude,
                                fold_topology = fold_topology)

            @test grid isa TripolarGrid

            @test grid.Nx == 4
            @test grid.Ny == 5
            @test grid.Nz == 1

            @test grid.conformal_mapping.first_pole_longitude == first_pole_longitude
            @test grid.conformal_mapping.north_poles_latitude == north_poles_latitude
            @test grid.conformal_mapping.southernmost_latitude == southernmost_latitude

            λᶜᶜᵃ = λnodes(grid, Center(), Center())
            φᶜᶜᵃ = φnodes(grid, Center(), Center())
            λᶠᶠᵃ = λnodes(grid, Face(), Face())
            φᶠᶠᵃ = φnodes(grid, Face(), Face())

            min_Δφ = @allowscalar minimum(φᶜᶜᵃ[:, 2] .- φᶜᶜᵃ[:, 1])
            @allowscalar begin
                # The tripolar grid should cover the whole longitude range
                # from the first_pole_longitude to first_pole_longitude + 360
                @test minimum(λᶜᶜᵃ) ≥ first_pole_longitude
                @test maximum(λᶜᶜᵃ) ≤ first_pole_longitude + 360
                @test minimum(λᶠᶠᵃ) ≥ first_pole_longitude
                @test maximum(λᶠᶠᵃ) ≤ first_pole_longitude + 360
                @test maximum(φᶜᶜᵃ) ≤ 90
                @test maximum(φᶠᶠᵃ) ≤ 90
                @test minimum(φᶜᶜᵃ) ≥ -90
                @test minimum(φᶠᶠᵃ) ≥ -90

                # The minimum latitude is not exactly the southernmost latitude because the grid
                # undulates slightly to maintain the same analytical description in the whole sphere
                # (i.e. constant latitude lines do not exist anywhere in this grid)
                @test minimum(φᶜᶜᵃ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southernmost_latitude
            end
        end
    end
end

@testset "Model tests..." begin
    for arch in archs
        @testset "$fold_topology fold topology" for fold_topology in fold_topologies
            grid = TripolarGrid(arch; size = (10, 10, 1), fold_topology = fold_topology)

            # Wrong free surface
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid)

            free_surface = SplitExplicitFreeSurface(grid; substeps = 12)
            model = HydrostaticFreeSurfaceModel(grid; free_surface)

            # Tests the grid has been extended
            η = model.free_surface.displacement
            P = model.free_surface.kernel_parameters

            range = contiguousrange(P)

            # Should have extended halos in the north
            Hx, Hy, _ = halo_size(η.grid)
            Nx, Ny, _ = size(grid)

            @test P isa KernelParameters
            @test range[1] == 1:Nx
            @test range[2] == 1:Ny+Hy-1

            @test Hx == halo_size(grid, 1)
            @test Hy != halo_size(grid, 2)
            @test Hy == length(free_surface.substepping.averaging_weights) + 2

            @test begin
                time_step!(model, 1.0)
                true
            end
        end
    end
end

@testset "Grid construction error tests..." begin
    for FT in float_types
        @testset "$fold_topology fold topology" for fold_topology in fold_topologies
            @test_throws ArgumentError TripolarGrid(CPU(), FT; size=(10, 10, 4), fold_topology = fold_topology, z=[-50.0, -30.0, -20.0, 0.0]) # too few z-faces
            @test_throws ArgumentError TripolarGrid(CPU(), FT; size=(10, 10, 4), fold_topology = fold_topology, z=[-2000.0, -1000.0, -50.0, -30.0, -20.0, 0.0]) # too many z-faces
        end
    end
end

@testset "Orthogonality of family of ellipses and hyperbolae..." begin
    for arch in archs
        @testset "$fold_topology fold topology" for fold_topology in fold_topologies
            # Test the orthogonality of a tripolar grid based on the orthogonality of a
            # cubed sphere of the same size (1ᵒ in latitude and longitude)
            cubed_sphere_grid = ConformalCubedSphereGrid(arch, panel_size = (90, 90, 1), z = (0, 1))
            cubed_sphere_panel = getregion(cubed_sphere_grid, 1)

            angle_cubed_sphere = on_architecture(arch, zeros(size(cubed_sphere_panel)...))
            cartesian_nodes, _ = get_cartesian_nodes_and_vertices(cubed_sphere_panel, Face(), Face(), Center())
            xF, yF, zF = cartesian_nodes
            xF = on_architecture(arch, xF)
            yF = on_architecture(arch, yF)
            zF = on_architecture(arch, zF)
            Nx, Ny, _  = size(cubed_sphere_panel)

            # Exclude the corners from the computation! (They are definitely not orthogonal)
            params = KernelParameters(5:Nx-5, 5:Ny-5)

            launch!(arch, cubed_sphere_panel, params, compute_nonorthogonality_angle!, angle_cubed_sphere, cubed_sphere_panel, xF, yF, zF)

            first_pole_longitude = λ¹ₚ = 75
            north_poles_latitude = φₚ  = 35

            λ²ₚ = λ¹ₚ + 180
            λ³ₚ = λ²ₚ + 180

            # Build a tripolar grid at 1ᵒ
            underlying_grid = TripolarGrid(arch; size = (360, 180, 1), first_pole_longitude, north_poles_latitude, fold_topology = fold_topology)

            # We need a bottom height field that ``masks'' the singularities
            bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                                ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                                ((abs(λ - λ³ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

            # Exclude the singularities from the computation! (They are definitely not orthogonal)
            tripolar_grid      = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
            angle_tripolar     = on_architecture(arch, zeros(size(tripolar_grid)...))
            cartesian_nodes, _ = get_cartesian_nodes_and_vertices(tripolar_grid.underlying_grid, Face(), Face(), Center())
            xF, yF, zF = cartesian_nodes
            xF = on_architecture(arch, xF)
            yF = on_architecture(arch, yF)
            zF = on_architecture(arch, zF)
            Nx, Ny, _  = size(tripolar_grid)

            launch!(arch, tripolar_grid, (Nx-1, Ny-1), compute_nonorthogonality_angle!, angle_tripolar, tripolar_grid, xF, yF, zF)

            @test maximum(angle_tripolar) < maximum(angle_cubed_sphere)
            @test minimum(angle_tripolar) > minimum(angle_cubed_sphere)
        end
    end
end

# We cannot rotate the entire grid because most of it is not symmetric around the pivot point,
# So here is a helper function for generating "valid" j indices around the pivot point of zipper.
# "valid" here meaning that the rotated and unrotated j indices remain within the interior + halo.
function pivotable_indices(jmin, jmax, jpivot)
    idx = jmin:jmax
    rotidx = Int.(2jpivot .- idx)
    valid = @. jmin ≤ rotidx ≤ jmax
    return idx[valid]
end

# Helper functions to test symmetry and antisymmetry with 180° rotation around the pivot point
isrot180symmetric(arr) = arr == rot180(arr)
isrot180antisymmetric(arr) = arr == -rot180(arr)

@testset "Zipper boundary conditions..." begin
    for arch in archs
        @testset "$fold_topology fold topology" for fold_topology in fold_topologies

            grid = TripolarGrid(arch; size = (10, 10, 1), fold_topology = fold_topology)
            Nx, Ny, _ = size(grid)
            Hx, Hy, _ = halo_size(grid)

            CC = CenterField(grid)
            FC = XFaceField(grid)
            CF = YFaceField(grid)
            FF = Field((Face(), Face(), Center()), grid)

            bcs = FieldBoundaryConditions()
            u_bcs = Oceananigans.BoundaryConditions.regularize_field_boundary_conditions(bcs, grid, :u)
            v_bcs = Oceananigans.BoundaryConditions.regularize_field_boundary_conditions(bcs, grid, :v)
            u = XFaceField(grid, boundary_conditions=u_bcs)
            v = YFaceField(grid, boundary_conditions=v_bcs)

            Pivot = (fold_topology == RightCenterFolded) ? UPivot : FPivot

            fields = (CC, FC, CF, FF, u, v)

            @testset "BC type" for f in fields
                @test f.boundary_conditions.north.classification isa Zipper{Pivot}
            end

            # The velocity fields are reversed at the north boundary
            # boundary_conditions.north.condition == -1, while the tracer
            # is not: boundary_conditions.north.condition == 1
            @testset "BC sign" begin
                @test CC.boundary_conditions.north.condition == 1
                @test FC.boundary_conditions.north.condition == 1
                @test CF.boundary_conditions.north.condition == 1
                @test FF.boundary_conditions.north.condition == 1
                @test u.boundary_conditions.north.condition == -1
                @test v.boundary_conditions.north.condition == -1
            end

            # set! random values then fill halos
            for f in fields
                set!(f, (x, y, z) -> rand())
                fill_halo_regions!(f)
            end

            # We use CPU architecture for scalar indexing.
            CC = on_architecture(CPU(), CC)
            CF = on_architecture(CPU(), CF)
            FC = on_architecture(CPU(), FC)
            FF = on_architecture(CPU(), FF)
            v = on_architecture(CPU(), v)
            u = on_architecture(CPU(), u)

            # Illustrated below are both cases with the pivot point (F or U) indicated.
            #          │           │           │           │           │           │           │
            # Ny+2 ─▶  ├──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼───  v ────┤
            #          │           │           │           │           │           │           │
            # Ny+1 ─▶  u     c     u     c     u     c     u     c     u     c     u     c     u
            #          │           │           │           │           │           │           │
            # Ny+1 ─▶  ├──── v ────┼──── v ────┼──── v ─── F ─── v ────┼──── v ────┼───  v ────┤ ◀─ Fold (RightFaceFolded)
            #          │           │           │           │           │           │           │
            #   Ny ─▶  u     c     u     c     u     c     U     c     u     c     u     c     u ◀─ Fold (RightCenterFolded)
            #          │           │           │           │           │           │           │
            #   Ny ─▶  ├──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┤
            #          │           │           │           │           │           │           │
            # Ny-1 ─▶  u     c     u     c     u     c     u     c     u     c     u     c     u
            #          │           │           │           │           │           │           │
            # Ny-1 ─▶  ├──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┤
            #          │           │           │           │           │           │           │
            #          ▲     ▲     ▲                       ▲                       ▲     ▲     ▲
            #          1     1     2                     Nx÷2+1                    Nx    Nx    Nx+1
            # For testing, rotate the entire grid around the central pivot point!

            # Use half-indices for the pivot-point index, which depends on the topology and location.
            # pivotjᶜ is the pivot index for center fields, and pivotjᶠ for face fields.
            pivotjᶜ, pivotjᶠ = (fold_topology == RightFaceFolded) ? (Ny + 1/2, Ny + 1) : (Ny, Ny + 1/2)

            # Then we take views centered around the pivot and rotate that view by 180°.
            # However we cannot rotate the entire grid and must restrict ourselves to those indices
            # that remain within the interior + halo after 180° rotation.
            maxjᶜ = Ny + Hy # max j for center fields
            maxjᶠ = Ny + Hy + (fold_topology == RightFaceFolded) # +1 for y-face fields if FPivot
            jᶜ = pivotable_indices(1 - Hy, maxjᶜ, pivotjᶜ)
            jᶠ = pivotable_indices(1 - Hy, maxjᶠ, pivotjᶠ)

            # Enforce zero velocities on the pivot points where u = -u and v = -v!
            # Only u velocity can be on pivot point for UPointPivot grid (RightCenterFolded)
            if fold_topology == RightCenterFolded
                u.data[[1, Nx ÷ 2 + 1, Nx + 1], pivotjᶜ, :] .= 0.0
            end

            # Test part of the halo with 180° rotation
            # (We cannot do it over all i indices because of the staggered grid)
            iᶜ = 1-Hx:Nx+Hx
            iᶠ = 1-Hx+1:Nx+Hx # <- skip the first (= westmost) index for rot180
            @testset "Test halo fill with rot180" begin
                @test isrot180symmetric(view(CC.data, iᶜ, jᶜ, 1))
                @test isrot180symmetric(view(FC.data, iᶠ, jᶜ, 1))
                @test isrot180symmetric(view(CF.data, iᶜ, jᶠ, 1))
                @test isrot180symmetric(view(FF.data, iᶠ, jᶠ, 1))
                @test isrot180antisymmetric(view(u.data, iᶠ, jᶜ, 1))
                @test isrot180antisymmetric(view(v.data, iᶜ, jᶠ, 1))
            end

            # Test over all i indices by applying reverse on each index and mod1 for i indices
            iᶜ = 1-Hx:Nx+Hx
            iᶜ′ = mod1.(reverse(iᶜ), Nx)
            iᶠ = 1-Hx:Nx+Hx
            iᶠ′ = mod1.(reverse(iᶠ) .+ 1, Nx)
            jᶜ′ = reverse(jᶜ)
            jᶠ′ = reverse(jᶠ)
            # Test that the northern halo region has been correctly rotated and sign-changed
            @testset "Test entire halo fill" begin
                @test view(CC.data, iᶜ, jᶜ, 1) ==  view(CC.data, iᶜ′, jᶜ′, 1)
                @test view(FC.data, iᶠ, jᶜ, 1) ==  view(FC.data, iᶠ′, jᶜ′, 1)
                @test view(CF.data, iᶜ, jᶠ, 1) ==  view(CF.data, iᶜ′, jᶠ′, 1)
                @test view(FF.data, iᶠ, jᶠ, 1) ==  view(FF.data, iᶠ′, jᶠ′, 1)
                @test view( u.data, iᶠ, jᶜ, 1) == -view( u.data, iᶠ′, jᶜ′, 1)
                @test view( v.data, iᶜ, jᶠ, 1) == -view( v.data, iᶜ′, jᶠ′, 1)
            end

            # Test that bottom height for an immersed boundary grid is also
            # correctly rotated and symmetric around the pivot point
            @testset "Test GridFittedBottom halo fill" begin
                grid = TripolarGrid(arch; size = (10, 10, 1), fold_topology = fold_topology)
                bottom(x, y) = rand()
                grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
                bottom_height = on_architecture(CPU(), grid.immersed_boundary.bottom_height.data)
                @test view(bottom_height, iᶜ, jᶜ, 1) == view(bottom_height, iᶜ′, jᶜ′, 1)
            end

        end
    end
end

using Oceananigans.Grids: with_halo, topology, halo_size

@testset "with_halo for TripolarGrid" begin
    for arch in archs
        @testset "$fold_topology fold topology [$arch]" for fold_topology in fold_topologies
            grid = TripolarGrid(arch; size = (20, 10, 4), z = (-100, 0), halo = (3, 3, 3),
                                fold_topology)

            new_grid = with_halo((5, 5, 5), grid)

            # Basic properties preserved
            @test new_grid.Nx == grid.Nx
            @test new_grid.Ny == grid.Ny
            @test new_grid.Nz == grid.Nz
            @test halo_size(new_grid) == (5, 5, 5)
            @test topology(new_grid) == topology(grid)
            @test new_grid.radius == grid.radius
            @test new_grid.conformal_mapping == grid.conformal_mapping

            Nx, Ny = grid.Nx, grid.Ny

            # Interior metrics (Δx, Δy, Az) and latitudes (φ) must be preserved exactly
            for (old_f, new_f) in [(grid.φᶜᶜᵃ, new_grid.φᶜᶜᵃ), (grid.φᶠᶜᵃ, new_grid.φᶠᶜᵃ),
                                   (grid.φᶜᶠᵃ, new_grid.φᶜᶠᵃ), (grid.φᶠᶠᵃ, new_grid.φᶠᶠᵃ),
                                   (grid.Δxᶜᶜᵃ, new_grid.Δxᶜᶜᵃ), (grid.Δxᶠᶜᵃ, new_grid.Δxᶠᶜᵃ),
                                   (grid.Δxᶜᶠᵃ, new_grid.Δxᶜᶠᵃ), (grid.Δxᶠᶠᵃ, new_grid.Δxᶠᶠᵃ),
                                   (grid.Δyᶜᶜᵃ, new_grid.Δyᶜᶜᵃ), (grid.Δyᶠᶜᵃ, new_grid.Δyᶠᶜᵃ),
                                   (grid.Δyᶜᶠᵃ, new_grid.Δyᶜᶠᵃ), (grid.Δyᶠᶠᵃ, new_grid.Δyᶠᶠᵃ),
                                   (grid.Azᶜᶜᵃ, new_grid.Azᶜᶜᵃ), (grid.Azᶠᶜᵃ, new_grid.Azᶠᶜᵃ),
                                   (grid.Azᶜᶠᵃ, new_grid.Azᶜᶠᵃ), (grid.Azᶠᶠᵃ, new_grid.Azᶠᶠᵃ)]
                old_cpu = on_architecture(CPU(), old_f)
                new_cpu = on_architecture(CPU(), new_f)
                @test all(old_cpu[1:Nx, 1:Ny] .== new_cpu[1:Nx, 1:Ny])
            end

            # Longitudes may differ by multiples of 180° at the fold row
            # (the fold BC can wrap longitudes or map to antipodal points at pivots)
            for (old_f, new_f) in [(grid.λᶜᶜᵃ, new_grid.λᶜᶜᵃ), (grid.λᶠᶜᵃ, new_grid.λᶠᶜᵃ),
                                   (grid.λᶜᶠᵃ, new_grid.λᶜᶠᵃ), (grid.λᶠᶠᵃ, new_grid.λᶠᶠᵃ)]
                old_cpu = on_architecture(CPU(), old_f)
                new_cpu = on_architecture(CPU(), new_f)
                diff = old_cpu[1:Nx, 1:Ny] .- new_cpu[1:Nx, 1:Ny]
                @test all(mod.(diff, 180) .≈ 0)
            end
        end
    end
end

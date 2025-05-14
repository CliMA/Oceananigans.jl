include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.BoundaryConditions: Zipper

using Oceananigans.Utils: KernelParameters
import Oceananigans.Utils: contiguousrange

contiguousrange(::KernelParameters{spec, offset}) where {spec, offset} = contiguousrange(spec, offset)

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
        grid = TripolarGrid(arch, size = (4, 5, 1), z = (0, 1),
                            first_pole_longitude = 75,
                            north_poles_latitude = 35,
                            southernmost_latitude = -80)

        @test grid isa TripolarGrid

        @test grid.Nx == 4
        @test grid.Ny == 5
        @test grid.Nz == 1

        @test grid.conformal_mapping.first_pole_longitude == 75
        @test grid.conformal_mapping.north_poles_latitude == 35
        @test grid.conformal_mapping.southernmost_latitude == -80

        λᶜᶜᵃ = λnodes(grid, Center(), Center())
        φᶜᶜᵃ = φnodes(grid, Center(), Center())

        min_Δφ = CUDA.@allowscalar minimum(φᶜᶜᵃ[:, 2] .- φᶜᶜᵃ[:, 1])

        @test minimum(λᶜᶜᵃ) ≥ 0
        @test maximum(λᶜᶜᵃ) ≤ 360
        @test maximum(φᶜᶜᵃ) ≤ 90

        # The minimum latitude is not exactly the southermost latitude because the grid
        # undulates slightly to maintain the same analytical description in the whole sphere
        # (i.e. constant latitude lines do not exist anywhere in this grid)
        @test minimum(φᶜᶜᵃ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southernmost_latitude
    end
end

@testset "Model tests..." begin
    for arch in archs
        grid = TripolarGrid(arch, size = (10, 10, 1))

        # Wrong free surface
        @test_throws ArgumentError HydrostaticFreeSurfaceModel(; grid)

        free_surface = SplitExplicitFreeSurface(grid; substeps = 12)
        model = HydrostaticFreeSurfaceModel(; grid, free_surface)

        # Tests the grid has been extended
        η = model.free_surface.η
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
        @test Hy == length(free_surface.substepping.averaging_weights) + 1

        @test begin
            time_step!(model, 1.0)
            true
        end
    end
end

@testset "Grid construction error tests..." begin
    for FT in float_types
        @test_throws ArgumentError TripolarGrid(CPU(), FT, size=(10, 10, 4), z=[-50.0, -30.0, -20.0, 0.0]) # too few z-faces
        @test_throws ArgumentError TripolarGrid(CPU(), FT, size=(10, 10, 4), z=[-2000.0, -1000.0, -50.0, -30.0, -20.0, 0.0]) # too many z-faces
    end
end

@testset "Orthogonality of family of ellipses and hyperbolae..." begin
    for arch in archs
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

        # Build a tripolar grid at 1ᵒ
        underlying_grid = TripolarGrid(arch; size = (360, 180, 1), first_pole_longitude, north_poles_latitude)

        # We need a bottom height field that ``masks'' the singularities
        bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                              ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

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

@testset "Zipper boundary conditions..." begin
    for arch in archs
        grid = TripolarGrid(arch; size = (10, 10, 1))
        Nx, Ny, _ = size(grid)
        Hx, Hy, _ = halo_size(grid)

        c = CenterField(grid)
        cx = XFaceField(grid)
        cy = YFaceField(grid)
        
        bcs = FieldBoundaryConditions()
        u_bcs = Oceananigans.BoundaryConditions.regularize_field_boundary_conditions(bcs, grid, :u)
        v_bcs = Oceananigans.BoundaryConditions.regularize_field_boundary_conditions(bcs, grid, :v)
        u = XFaceField(grid, boundary_conditions=u_bcs)
        v = YFaceField(grid, boundary_conditions=v_bcs)

        @test c.boundary_conditions.north.classification isa Zipper
        @test cx.boundary_conditions.north.classification isa Zipper
        @test cy.boundary_conditions.north.classification isa Zipper
        @test u.boundary_conditions.north.classification isa Zipper
        @test v.boundary_conditions.north.classification isa Zipper

        # The velocity fields are reversed at the north boundary
        # boundary_conditions.north.condition == -1, while the tracer
        # is not: boundary_conditions.north.condition == 1
        @test c.boundary_conditions.north.condition == 1
        @test cx.boundary_conditions.north.condition == 1
        @test cy.boundary_conditions.north.condition == 1
        @test u.boundary_conditions.north.condition == -1
        @test v.boundary_conditions.north.condition == -1

        set!(c, 1)
        set!(cx, 1)
        set!(cy, 1)
        set!(u, 1)
        set!(v, 1)

        fill_halo_regions!(c)
        fill_halo_regions!(cx)
        fill_halo_regions!(cy)
        fill_halo_regions!(u)   
        fill_halo_regions!(v)

        north_boundary_c = on_architecture(CPU(), view(c.data, :, Ny+1:Ny+Hy, 1))
        north_boundary_cy = on_architecture(CPU(), view(cy.data, :, Ny+1:Ny+Hy, 1))
        north_boundary_v = on_architecture(CPU(), view(v.data, :, Ny+1:Ny+Hy, 1))
        @test all(north_boundary_c .== 1)
        @test all(north_boundary_cy .== 1)
        @test all(north_boundary_v .== -1)

        north_interior_boundary_cx = on_architecture(CPU(), view(cx.data, 2:Nx-1, Ny+1:Ny+Hy, 1))
        north_interior_boundary_u = on_architecture(CPU(), view(u.data, 2:Nx-1, Ny+1:Ny+Hy, 1))
        @test all(north_interior_boundary_cx .== 1)
        @test all(north_interior_boundary_u .== -1)

        north_boundary_u_left  = on_architecture(CPU(), view(u.data, 1, Ny+1:Ny+Hy, 1))
        north_boundary_u_right = on_architecture(CPU(), view(u.data, Nx+1, Ny+1:Ny+Hy, 1))
        @test all(north_boundary_u_left  .== 1)
        @test all(north_boundary_u_right .== 1)

        grid = TripolarGrid(arch; size = (10, 10, 1))
        bottom(x, y) = rand()
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
        bottom_height = grid.immersed_boundary.bottom_height

        @test on_architecture(CPU(), interior(bottom_height, :, 10, 1)) == on_architecture(CPU(), interior(bottom_height, 10:-1:1, 10, 1))

        c = CenterField(grid)
        cx = XFaceField(grid)
        u = XFaceField(grid, boundary_conditions=u_bcs)

        set!(c, (x, y, z) -> x)
        set!(cx, (x, y, z) -> x)
        set!(u, (x, y, z) -> x)

        fill_halo_regions!(c)
        fill_halo_regions!(cx)
        fill_halo_regions!(u)

        @test on_architecture(CPU(), interior(c, :, 10, 1)) == on_architecture(CPU(), interior(c, 10:-1:1, 10, 1))
        # For x face fields the first element is unique and we remove the
        # north pole that is exactly at Nx+1
        left_side  = on_architecture(CPU(), interior(cx, 2:5, 10, 1))
        right_side = on_architecture(CPU(), interior(cx, 7:10, 10, 1))

        # The sign of auxiliary XFaceField are _not_ reversed at the north boundary
        @test left_side == reverse(right_side)

        left_side  = on_architecture(CPU(), interior(u, 2:5, 10, 1))
        right_side = on_architecture(CPU(), interior(u, 7:10, 10, 1))

        # The sign of `u` is reversed at the north boundary
        @test left_side == - reverse(right_side)

    end
end

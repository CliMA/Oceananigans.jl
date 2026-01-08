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
    for arch in archs, fold_topology in fold_topologies
        grid = TripolarGrid(arch;
                            size = (4, 5, 1),
                            z = (0, 1),
                            first_pole_longitude = 75,
                            north_poles_latitude = 35,
                            southernmost_latitude = -80,
                            fold_topology = fold_topology)

        @test grid isa TripolarGrid

        @test grid.Nx == 4
        @test grid.Ny == 5
        @test grid.Nz == 1

        @test grid.conformal_mapping.first_pole_longitude == 75
        @test grid.conformal_mapping.north_poles_latitude == 35
        @test grid.conformal_mapping.southernmost_latitude == -80

        λᶜᶜᵃ = λnodes(grid, Center(), Center())
        φᶜᶜᵃ = φnodes(grid, Center(), Center())

        min_Δφ = @allowscalar minimum(φᶜᶜᵃ[:, 2] .- φᶜᶜᵃ[:, 1])
        @allowscalar begin
            @test minimum(λᶜᶜᵃ) ≥ 0
            @test maximum(λᶜᶜᵃ) ≤ 360
            @test maximum(φᶜᶜᵃ) ≤ 90

            # The minimum latitude is not exactly the southermost latitude because the grid
            # undulates slightly to maintain the same analytical description in the whole sphere
            # (i.e. constant latitude lines do not exist anywhere in this grid)
            @test minimum(φᶜᶜᵃ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southernmost_latitude
        end
    end
end

@testset "Model tests..." begin
    for arch in archs, fold_topology in fold_topologies
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
        @test range[2] == 1:(Ny + Hy - 1 + (fold_topology == RightFaceFolded))

        @test Hx == halo_size(grid, 1)
        @test Hy != halo_size(grid, 2)
        @test Hy == length(free_surface.substepping.averaging_weights) + 2

        @test begin
            time_step!(model, 1.0)
            true
        end
    end
end

@testset "Grid construction error tests..." begin
    for FT in float_types, fold_topology in fold_topologies
        @test_throws ArgumentError TripolarGrid(CPU(), FT; size=(10, 10, 4), fold_topology = fold_topology, z=[-50.0, -30.0, -20.0, 0.0]) # too few z-faces
        @test_throws ArgumentError TripolarGrid(CPU(), FT; size=(10, 10, 4), fold_topology = fold_topology, z=[-2000.0, -1000.0, -50.0, -30.0, -20.0, 0.0]) # too many z-faces
    end
end

@testset "Orthogonality of family of ellipses and hyperbolae..." begin
    for arch in archs, fold_topology in fold_topologies
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
        underlying_grid = TripolarGrid(arch; size = (360, 180, 1), first_pole_longitude, north_poles_latitude, fold_topology = fold_topology)

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

# helper function for generating indices around the pivot point of zipper
function pivoted_indices(idxmin, idxmax, idxpivot)
    idx = idxmin:idxmax
    rotidx = Int.(2idxpivot .- idx)
    valid = @. idxmin ≤ rotidx ≤ idxmax
    return idx[valid], rotidx[valid]
end

@testset "Zipper boundary conditions..." begin
    for arch in archs, fold_topology in fold_topologies
        grid = TripolarGrid(arch; size = (10, 10, 1), fold_topology = fold_topology)
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

        Pivot = (fold_topology == RightCenterFolded) ? UPivot : FPivot

        @test c.boundary_conditions.north.classification isa Zipper{Pivot}
        @test cx.boundary_conditions.north.classification isa Zipper{Pivot}
        @test cy.boundary_conditions.north.classification isa Zipper{Pivot}
        @test u.boundary_conditions.north.classification isa Zipper{Pivot}
        @test v.boundary_conditions.north.classification isa Zipper{Pivot}

        # The velocity fields are reversed at the north boundary
        # boundary_conditions.north.condition == -1, while the tracer
        # is not: boundary_conditions.north.condition == 1
        @test c.boundary_conditions.north.condition == 1
        @test cx.boundary_conditions.north.condition == 1
        @test cy.boundary_conditions.north.condition == 1
        @test u.boundary_conditions.north.condition == -1
        @test v.boundary_conditions.north.condition == -1

        set!(c, (x, y, z) -> x + y + z)
        set!(cx, (x, y, z) -> x + y + z)
        set!(cy, (x, y, z) -> x + y + z)
        set!(u, (x, y, z) -> x + y + z)
        set!(v, (x, y, z) -> x + y + z)

        fill_halo_regions!(c)
        fill_halo_regions!(cx)
        fill_halo_regions!(cy)
        fill_halo_regions!(u)
        fill_halo_regions!(v)

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
        # For testing, we define all the indices by symmetry around the pivot point!
        # The pivot-point indices are referenced to the Center locations (hence the half indices).
        c_pivot_i = v_pivot_i = Nx ÷ 2 + 0.5
        u_pivot_i = Nx ÷ 2 + 1
        c_pivot_j = u_pivot_j = (fold_topology == RightCenterFolded) ? Ny : Ny + 0.5
        v_pivot_j = c_pivot_j + 0.5
        # We will take views centered around the pivot
        imin, imax = 1 - Hx, Nx + Hx
        jmin = 1 - Hy
        u_jmax = c_jmax = Ny + Hy
        v_jmax = (fold_topology == RightCenterFolded) ? (Ny + Hy) : (Ny + Hy + 1)
        c_i, c_i′ = pivoted_indices(imin, imax, c_pivot_i)
        c_j, c_j′ = pivoted_indices(jmin, c_jmax, c_pivot_j)
        u_i, u_i′ = pivoted_indices(imin, imax, u_pivot_i)
        u_j, u_j′ = pivoted_indices(jmin, u_jmax, u_pivot_j)
        v_i, v_i′ = pivoted_indices(imin, imax, v_pivot_i)
        v_j, v_j′ = pivoted_indices(jmin, v_jmax, v_pivot_j)

        # Test that the northern halo region has been correctly rotated and sign-changed
        c = on_architecture(CPU(), c)
        cy = on_architecture(CPU(), cy)
        v = on_architecture(CPU(), v)
        cx = on_architecture(CPU(), cx)
        u = on_architecture(CPU(), u)
        # Before we run the tests, enforce zero velocities on the pivot points!
        # Only u can be on pivot point for UPointPivot grid (RightCenterFolded)
        # Maybe this can be avoided with some land over the pivot points?
        if fold_topology == RightCenterFolded
            u.data[[1, u_pivot_i, Nx + 1], u_pivot_j, :] .= 0.0
        end
        @test all(view(c.data, c_i, c_j, 1) .== view(c.data, c_i′, c_j′, 1))
        @test all(view(cy.data, v_i, v_j, 1) .== view(cy.data, v_i′, v_j′, 1))
        @test all(view(v.data, v_i, v_j, 1) .== -view(v.data, v_i′, v_j′, 1))
        @test all(view(cx.data, u_i, u_j, 1) .== view(cx.data, u_i′, u_j′, 1))
        @test all(view(u.data, u_i, u_j, 1) .== -view(u.data, u_i′, u_j′, 1))

        grid = TripolarGrid(arch; size = (10, 10, 1), fold_topology = fold_topology)
        bottom(x, y) = rand()
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
        bottom_height = grid.immersed_boundary.bottom_height

        @test on_architecture(CPU(), view(bottom_height.data, c_i, c_j, 1)) == on_architecture(CPU(), view(bottom_height.data, c_i′, c_j′, 1))

    end
end

isdefined(@__MODULE__, :archs) || include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Test
using Adapt
using Oceananigans
using Oceananigans.Grids: SphericalGrid, EquiangularGnomonicPanelGrid, EquiangularGnomonicMapping, ThinShell, DeepShell,
                          isorthogonal, equiangular_gnomonic_map, inverse_equiangular_gnomonic_map, longitude_latitude,
                          spherical_unit_vector, λnodes, φnodes, Connected, Sides
using Oceananigans.Operators: Azᶜᶜᵃ, Δxᶜᶠᵃ, cosαᶠᶠᵃ, sinαᶠᶠᵃ, cosαᶜᶜᵃ, sinαᶜᶜᵃ
using Oceananigans.OrthogonalSphericalShellGrids: ConformalCubedSpherePanelGrid
using Oceananigans.MultiRegion: EquiangularCubedSphereGrid, SphericalCubedSphereGrid, CubedSphereConnectivity,
                                CubedSpherePartition
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: getregion
using Oceananigans.Architectures: on_architecture

const metric_names = (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
                      :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
                      :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)

const angle_names = (:cosαᶜᶜᵃ, :cosαᶠᶜᵃ, :cosαᶜᶠᵃ, :cosαᶠᶠᵃ, :sinαᶜᶜᵃ, :sinαᶠᶜᵃ, :sinαᶜᶠᵃ, :sinαᶠᶠᵃ)

function throws_orthogonality_error(f)
    try
        f()
    catch error
        return error isa ArgumentError && occursin("orthogonal", sprint(showerror, error))
    end
    return false
end

@testset "SphericalGrid" begin
    R = 6.3712e6

    @testset "Equiangular gnomonic map and its inverse" begin
        rotation = CubedSphereConnectivity(CubedSpherePartition(R = 1)).rotations[4]
        for (ξ, η) in ((0.0, 0.0), (0.3, -0.7), (-π/4, π/4), (0.9, 1.1))
            for rot in (nothing, rotation)
                v = equiangular_gnomonic_map(ξ, η, rot)
                @test sum(abs2, v) ≈ 1
                ξ′, η′ = inverse_equiangular_gnomonic_map(v, rot)
                @test ξ′ ≈ ξ atol = 4eps()
                @test η′ ≈ η atol = 4eps()
            end
        end
        @test longitude_latitude(equiangular_gnomonic_map(0.0, 0.0))[2] == 90
    end

    @testset "Single equiangular panel (N = $N)" for N in (15, 16)
        grid = EquiangularGnomonicPanelGrid(CPU(), Float64; size = (N, N, 2), z = (0, 1), halo = (3, 3, 1), radius = R)

        @test grid isa SphericalGrid
        @test !isorthogonal(grid)
        @test grid.mapping isa EquiangularGnomonicMapping
        @test grid.shell isa ThinShell

        for name in (metric_names..., angle_names...)
            @test all(isfinite, parent(getproperty(grid, name)))
        end

        # The panel is one sixth of the sphere
        @test sum(view(grid.Azᶜᶜᵃ, 1:N, 1:N)) ≈ 4π * R^2 / 6 rtol = 1e-13
        @test sum(Azᶜᶜᵃ(i, j, 1, grid) for i in 1:N, j in 1:N) ≈ 4π * R^2 / 6 rtol = 1e-13

        # The coordinate lines meet at right angles at the panel centre ...
        if iseven(N)
            c = N ÷ 2 + 1
            @test abs(cosαᶠᶠᵃ(c, c, 1, grid)) < 1e-14
            @test sinαᶠᶠᵃ(c, c, 1, grid) ≈ 1 atol = 1e-15
        else
            c = (N + 1) ÷ 2
            @test abs(cosαᶜᶜᵃ(c, c, 1, grid)) < 1e-14
            @test sinαᶜᶜᵃ(c, c, 1, grid) ≈ 1 atol = 1e-15
        end

        # ... and at 120° (60° between an inward and an outward edge direction) at the cube vertices
        @test grid.cosαᶠᶠᵃ[1, 1]     ≈ -1/2 atol = 1e-13
        @test grid.cosαᶠᶠᵃ[N+1, 1]   ≈ +1/2 atol = 1e-13
        @test grid.cosαᶠᶠᵃ[N+1, N+1] ≈ -1/2 atol = 1e-13
        @test grid.cosαᶠᶠᵃ[1, N+1]   ≈ +1/2 atol = 1e-13
        @test all(>(0), parent(grid.sinαᶜᶜᵃ))

        # Non-orthogonal everywhere except on the panel's centre lines
        @test maximum(abs, view(grid.cosαᶜᶜᵃ, 1:N, 1:N)) > 0.3

        # Edge lengths are great-circle distances between corners
        P(i, j) = spherical_unit_vector(grid.λᶠᶠᵃ[i, j], grid.φᶠᶠᵃ[i, j])
        @test Δxᶜᶠᵃ(3, 4, 1, grid) ≈ R * acos(sum(P(3, 4) .* P(4, 4))) rtol = 1e-10

        # Same storage layout as ConformalCubedSpherePanelGrid with the same size, halo and topology
        conformal = ConformalCubedSpherePanelGrid(CPU(), Float64; size = (N, N, 2), z = (0, 1), halo = (3, 3, 1),
                                                  topology = (Bounded, Bounded, Bounded))
        for name in metric_names
            @test axes(getproperty(grid, name)) == axes(getproperty(conformal, name))
        end

        @test size(λnodes(grid, Center(), Center())) == (N, N)
        @test size(φnodes(grid, Face(), Face())) == (N + 1, N + 1)
        @test occursin("SphericalGrid", sprint(show, grid))

        cpu_grid = on_architecture(CPU(), grid)
        @test cpu_grid.Azᶜᶜᵃ == grid.Azᶜᶜᵃ
        adapted = Adapt.adapt_structure(Array, grid)
        @test adapted.cosαᶠᶜᵃ == grid.cosαᶠᶜᵃ
        @test isnothing(adapted.architecture)

        # Orthogonal-only code paths refuse the grid
        @test throws_orthogonality_error(() -> NonhydrostaticModel(grid))
        @test throws_orthogonality_error(() -> HydrostaticFreeSurfaceModel(grid))
        @test throws_orthogonality_error(() -> Oceananigans.Operators.intrinsic_vector(1, 1, 1, grid, 1.0, 0.0))

        # Negative control: an orthogonal grid passes the same guard
        rectilinear = RectilinearGrid(CPU(); size = (2, 2, 2), extent = (1, 1, 1))
        @test isorthogonal(rectilinear)
        @test !throws_orthogonality_error(() -> NonhydrostaticModel(rectilinear))

        field = CenterField(grid)
        set!(field, (λ, φ, z) -> φ)
        @test interior(field, :, :, 1) == view(grid.φᶜᶜᵃ, 1:N, 1:N)
    end

    @testset "Constructor options" begin
        @test_throws ArgumentError SphericalGrid(CPU(); size = (4, 4, 1), z = (0, 1), shell = DeepShell())
        @test_throws ArgumentError EquiangularGnomonicPanelGrid(CPU(); size = (4, 4, 1), z = (0, 1), halo = (2, 2, 1))
        empty_grid = SphericalGrid(CPU(), Float64; size = (4, 4, 1), z = (0, 1), halo = (2, 2, 1))
        @test all(isnan, parent(empty_grid.Δxᶠᶜᵃ))
        @test_throws ArgumentError SphericalGrid(CPU(), Float64; size = (4, 4, 1), z = (0, 1), halo = (2, 2, 1),
                                                 vertices = (zeros(5, 5), zeros(5, 5)))
    end

    @testset "Six-panel equiangular cubed sphere" begin
        N, H = 12, 3
        grid = EquiangularCubedSphereGrid(CPU(), Float64; panel_size = (N, N, 2), z = (0, 1),
                                          horizontal_direction_halo = H, z_halo = 1, radius = R)

        @test !isorthogonal(grid)
        @test sum(sum(view(getregion(grid, r).Azᶜᶜᵃ, 1:N, 1:N)) for r in 1:6) ≈ 4π * R^2 rtol = 1e-13

        # Halo values filled from the neighbouring panels are the nearby centres: the panels' orientations
        # agree with CubedSphereConnectivity
        centre(panel, i, j) = spherical_unit_vector(panel.λᶜᶜᵃ[i, j], panel.φᶜᶜᵃ[i, j])
        components = ntuple(_ -> CenterField(grid), 3)
        for r in 1:6, (d, c) in enumerate(components)
            panel = getregion(grid, r)
            data = getregion(c, r)
            for k in 1:2, j in 1:N, i in 1:N
                data[i, j, k] = centre(panel, i, j)[d]
            end
        end
        foreach(fill_halo_regions!, components)

        # The neighbour's coordinate lines kink at the seam, so the panel's continued halo centres drift away from
        # the neighbour's centres with depth; at depth one the filled centre must be the nearest one (< Δσ / 2).
        Δσ = (π / 2) / N
        worst = zeros(H)
        for r in 1:6
            panel = getregion(grid, r)
            for j in 1-H:N+H, i in 1-H:N+H
                xor(1 <= i <= N, 1 <= j <= N) || continue
                depth = max(1 - i, i - N, 1 - j, j - N)
                filled = ntuple(d -> getregion(components[d], r)[i, j, 1], 3)
                continued = centre(panel, i, j)
                worst[depth] = max(worst[depth], acos(clamp(sum(filled .* continued), -1, 1)))
            end
        end
        @test worst[1] < Δσ / 2
        @test all(worst .< H * Δσ)

        # Panel corners and centres coincide with those of ConformalCubedSphereGrid (same CubedSphereConnectivity)
        conformal = ConformalCubedSphereGrid(CPU(), Float64; panel_size = (N, N, 2), z = (0, 1),
                                             horizontal_direction_halo = H, z_halo = 1, radius = R)
        for r in 1:6, (i, j) in ((1, 1), (N+1, 1), (N+1, N+1), (1, N+1), (N÷2 + 1, N÷2 + 1))
            a = spherical_unit_vector(getregion(grid, r).λᶠᶠᵃ[i, j], getregion(grid, r).φᶠᶠᵃ[i, j])
            b = spherical_unit_vector(getregion(conformal, r).λᶠᶠᵃ[i, j], getregion(conformal, r).φᶠᶠᵃ[i, j])
            @test maximum(abs, a .- b) < 1e-12
        end

        # Negative control: a panel built with another panel's rotation is detected by the same check
        rotations = CubedSphereConnectivity(CubedSpherePartition(R = 1)).rotations
        misplaced = EquiangularGnomonicPanelGrid(CPU(), Float64; size = (N, N, 2), z = (0, 1), halo = (H, H, 1),
                                                 radius = R, rotation = rotations[2])
        a = spherical_unit_vector(misplaced.λᶠᶠᵃ[1, 1], misplaced.φᶠᶠᵃ[1, 1])
        b = spherical_unit_vector(getregion(conformal, 1).λᶠᶠᵃ[1, 1], getregion(conformal, 1).φᶠᶠᵃ[1, 1])
        @test maximum(abs, a .- b) > 0.1

        composed = SphericalCubedSphereGrid(Tuple(getregion(grid, r) for r in 1:6))
        @test composed.region_grids[3].Azᶜᶜᵃ == getregion(grid, 3).Azᶜᶜᵃ
    end
end

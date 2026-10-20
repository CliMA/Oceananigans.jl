include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: with_halo
using Oceananigans.MultiRegion: YinYangField, default_yin_yang_overlap, yin_yang_panel_extents, yin_yang_transform,
                                yin_yang_panel_face_coordinates, yin_yang_rotation_angle, yin_yang_masks, yin_yang_owned,
                                yin_yang_cell_weights, owned_weights, dividing_curve_latitude
using Oceananigans.Operators: rotation_angle
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

# The largest distance (degrees) between the geographic points (λ, φ) and (λ′, φ′)
angular_distance((λ, φ), (λ′, φ′)) = max(abs(wrap_longitude(λ - λ′)) * cosd(φ), abs(φ - φ′))

# Maximum difference (degrees) at cell centers between the closed-form rotation angle and the finite-difference
# angle of `Operators.rotation_angle`, away from the geographic poles where the geographic basis is singular
function maximum_rotation_angle_error(grid)
    centers = filter(ij -> abs(node_coordinates(ij..., grid, Center(), Center())[2]) ≤ 80, interior_nodes(grid))
    Δθ(i, j) = rotation_angle(i, j, grid) - rotation_angle(i, j, grid, Center(), Center())
    return maximum(abs(wrap_longitude(rad2deg(Δθ(i, j)))) for (i, j) in centers)
end

cell_areas(λᶠ, φᶠ) = [deg2rad(λᶠ[i+1] - λᶠ[i]) * (sind(φᶠ[j+1]) - sind(φᶠ[j]))
                      for i in 1:length(λᶠ)-1, j in 1:length(φᶠ)-1]

# Latitude (degrees) of the dividing curve above longitude λ, by bisection of the ownership rule
function bisected_dividing_curve_latitude(λ)
    lower, upper = 0.0, 45.0
    for _ in 1:100
        middle = (lower + upper) / 2
        yin_yang_owned(λ, middle) ? (lower = middle) : (upper = middle)
    end
    return (lower + upper) / 2
end

tilted_grid(N) = RotatedLatitudeLongitudeGrid(CPU(); size=(5N, 4N, 1), longitude=(-100, 60), latitude=(-50, 70),
                                              north_pole=(70, 55), z=(-1, 0))

cpu_grid = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0))
yin, yang = cpu_grid[1], cpu_grid[2]
Nx, Ny, _ = size(yin)
Hx, Hy, _ = halo_size(yin)

@testset "YinYangGrid construction and owned weights [$(typeof(arch)), $FT]" for arch in archs, FT in float_types
    grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))
    δ = default_yin_yang_overlap((3, 3, 3))
    λᵉ, φᵉ = yin_yang_panel_extents(grid.connectivity.core_size, δ)

    @test eltype(grid) == FT
    @test topology(grid) == (FullyConnected, FullyConnected, Bounded)
    @test δ == grid.connectivity.overlap == 3
    @test size(grid) == (48 + 2δ, 16 + 2δ, 2)
    @test all((λᵉ, φᵉ) .≈ (135 + δ * 90 / 16, 45 + δ * 90 / 16))
    @test grid[1] isa RotatedLatitudeLongitudeGrid
    @test typeof(grid[1]) == typeof(grid[2])
    @test CenterField(grid) isa YinYangField
    @test summary(grid) ==
        "54×22×2 YinYangGrid{$FT, FullyConnected, FullyConnected, Bounded} on $(summary(arch)) with 3×3×3 halo"
    @test occursin("on every side, 4×4 interpolation stencils", sprint(show, grid))
    @test summary(grid.connectivity) == "YinYangConnectivity with 48×16 core, 3-cell overlap and 4×4 interpolation stencils"
    @test_throws ArgumentError YinYangGrid(arch, FT; size=(40, 16, 2), z=(-1, 0))
    @test_throws ArgumentError YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0), overlap=5)

    w = owned_weights(grid)
    @test w isa YinYangField
    @test location(w) == (Center, Center, Nothing)
    @test getregion(w, 1).data === getregion(w, 2).data
    weights = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)
    @test eltype(weights) == FT
    @test axes(weights) == (1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:1)
    @test all(iszero, weights[Nx+1:Nx+Hx, :, 1])

    # The panel area is analytic, and the owned cells of the two panels tile the sphere, to rounding error
    R = grid[1].radius
    rtol = FT == Float64 ? 1e-12 : 1e-4
    Az = on_architecture(CPU(), grid[1].Azᶜᶜᵃ)
    @test sum(Az[1:Nx, 1:Ny]) ≈ R^2 * deg2rad(2λᵉ) * 2sind(φᵉ) rtol=rtol
    @test 2 * sum(Float64(weights[i, j, 1]) * Az[i, j] for i in 1:Nx, j in 1:Ny) ≈ 4π * R^2 rtol=rtol

    wide_grid = with_halo((4, 4, 4), grid)
    @test (halo_size(wide_grid), size(wide_grid), wide_grid.connectivity.overlap) == ((4, 4, 4), size(grid), δ)
    @test axes(yin_yang_masks(wide_grid).cell_weights) == (-3:Nx+4, -3:Ny+4, 1:1)

    # The panel coordinates are computed in the grid's precision, and agree with Float64 to that precision
    typed_yang = on_architecture(CPU(), grid[2])
    @test eltype(typed_yang.λᶜᶜᵃ) == FT
    @test maximum(abs, typed_yang.φᶜᶜᵃ[1:Nx, 1:Ny] .- yang.φᶜᶜᵃ[1:Nx, 1:Ny]) < (FT == Float64 ? 1e-12 : 1e-3)
end

@testset "Overlap validation" begin
    # At 2° the binding receivers sit exactly on donor centers and need δ ≥ 1 for 4×4 stencils
    @test_throws ArgumentError YinYangGrid(CPU(); size=(135, 45, 1), z=(-1, 0), overlap=0)
    @test YinYangGrid(CPU(); size=(135, 45, 1), z=(-1, 0), overlap=1) isa YinYangGrid
    @test default_yin_yang_overlap((5, 5, 5)) == 4
end

@testset "Kageyama-Sato involution" begin
    Random.seed!(0)
    points = [(360rand() - 180, asind(2rand() - 1)) for _ in 1:200]
    @test maximum(p -> angular_distance(p, yin_yang_transform(yin_yang_transform(p...)...)), points) < 1e-10
    @test all(yin_yang_transform(0, 0) .≈ (180, 0))

    # The panel coordinates of node (i, j) on Yang are the geographic coordinates of node (i, j) on Yin,
    # so the geographic coordinates of Yang's nodes are the involution of Yin's, halos included
    error = 0.0
    for (ℓx, ℓy) in yin_yang_locations, j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx
        λ′φ′ = yin_yang_transform(node_coordinates(i, j, yin, ℓx, ℓy)...)
        error = max(error, angular_distance(node_coordinates(i, j, yang, ℓx, ℓy), λ′φ′))
    end
    @test error < 1e-10

    # The face-face node in the middle of Yang lies at (180°, 0), and Yang's grid north pole is at (90°, 0)
    i₀, j₀ = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    @test node_coordinates(i₀, j₀, yin, Face(), Face()) == (0, 0)
    @test angular_distance(node_coordinates(i₀, j₀, yang, Face(), Face()), (180, 0)) < 1e-10
    @test all(north_pole(yang) .== (90, 0))
end

@testset "Rotation angles" begin
    # Yin's panel frame is the geographic frame
    @test all(abs(rotation_angle(i, j, yin, ℓx, ℓy)) < 1e-12
              for (ℓx, ℓy) in yin_yang_locations, (i, j) in interior_nodes(yin))

    # The closed-form angle converges to the finite-difference angle of `Operators.rotation_angle`, on Yang and on
    # a tilted RotatedLatitudeLongitudeGrid with the geographic south pole near its southern edge
    @test maximum_rotation_angle_error(yang) < 1
    coarse_error, fine_error = (maximum_rotation_angle_error(tilted_grid(N)) for N in (4, 16))
    @test fine_error < 0.5
    @test fine_error < coarse_error / 10
    @test abs(rotated_basis_angle(10, 20, 0, 90)) < 1e-15

    # The Yin-to-Yang angle at a geographic point is the rotation angle of the Yang grid there
    error = 0.0
    for (ℓx, ℓy) in yin_yang_locations, j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx
        λ, φ = node_coordinates(i, j, yang, ℓx, ℓy)
        error = max(error, abs(yin_yang_rotation_angle(λ, φ) - rotation_angle(i, j, yang, ℓx, ℓy)))
    end
    @test error < 1e-12

    # The angle at the same point in the other panel's frame is -ψ, and ψ rotates the Cartesian basis vectors
    # of the geographic frame onto those of Yang's frame, mapped by M to the geographic frame
    M = [-1 0 0; 0 0 1; 0 1 0]
    east(λ) = [-sind(λ), cosd(λ), 0]
    north(λ, φ) = [-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ)]
    Random.seed!(1234)
    for _ in 1:20
        λ, φ = 360rand() - 180, 170rand() - 85
        λ′, φ′ = yin_yang_transform(λ, φ)
        ψ, ψ′ = yin_yang_rotation_angle(λ, φ), yin_yang_rotation_angle(λ′, φ′)
        @test abs(sin(ψ + ψ′)) < 1e-10
        @test cos(ψ + ψ′) ≈ 1
        @test cos(ψ) ≈ east(λ) ⋅ (M * east(λ′))
        @test sin(ψ) ≈ east(λ) ⋅ (M * north(λ′, φ′))
    end
end

@testset "Ownership and owned area fractions" begin
    Random.seed!(1)
    points = [(360rand() - 180, asind(2rand() - 1)) for _ in 1:2000]
    inside_core((λ, φ)) = abs(λ) ≤ 135 && abs(φ) ≤ 45
    owners = [(yin_yang_owned(p...), yin_yang_owned(yin_yang_transform(p...)...)) for p in points]
    @test all(sum(o) == 1 for o in owners)
    @test all(inside_core(o[1] ? p : yin_yang_transform(p...)) for (p, o) in zip(points, owners))

    # The closed-form dividing curve agrees with the ownership rule
    λs = range(0, 134.99, length=200)
    @test rad2deg.(dividing_curve_latitude.(deg2rad.(λs))) ≈ bisected_dividing_curve_latitude.(λs) rtol=1e-12

    for (Nφ, overlap) in ((45, 2), (9, 0))
        λᶠ, φᶠ = yin_yang_panel_face_coordinates((3Nφ, Nφ), overlap)
        w = yin_yang_cell_weights(λᶠ, φᶠ)
        @test all(0 .≤ w .≤ 1)
        @test sum(w .* cell_areas(λᶠ, φᶠ)) ≈ 2π rtol=1e-12
        @test w ≈ reverse(w, dims=1) rtol=1e-14
        @test w ≈ reverse(w, dims=2) rtol=1e-14
    end

    # The owned fractions of cut cells match a midpoint rule on the ownership rule, which converges like h^1.5
    λᶠ, φᶠ = yin_yang_panel_face_coordinates((135, 45), 0)
    w = yin_yang_cell_weights(λᶠ, φᶠ)
    areas = cell_areas(λᶠ, φᶠ)
    errors = map(findall(0 .< w .< 1)) do I
        i, j = Tuple(I)
        samples = range(λᶠ[i], λᶠ[i+1], length=401)
        φ★ = deg2rad.(bisected_dividing_curve_latitude.((samples[1:end-1] .+ samples[2:end]) ./ 2))
        φ₁, φ₂ = deg2rad(φᶠ[j]), deg2rad(φᶠ[j+1])
        owned_sines = sin.(clamp.(φ★, φ₁, φ₂)) .- sin.(clamp.(-φ★, φ₁, φ₂))
        owned_area = sum(owned_sines) * deg2rad(λᶠ[i+1] - λᶠ[i]) / 400
        abs(w[i, j] - owned_area / areas[i, j])
    end
    @test maximum(errors) < 1e-3
end

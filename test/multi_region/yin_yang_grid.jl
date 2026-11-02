include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: with_halo
using Oceananigans.MultiRegion: YinYangField, yin_yang_transform, yin_yang_panel_face_coordinates, yin_yang_rotation_angle,
                                yin_yang_masks, yin_yang_owned, owned_weights, dividing_curve_latitude
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

# Latitude (degrees) of the dividing curve above longitude λ, by bisection of the ownership rule
function bisected_dividing_curve_latitude(λ)
    lower, upper = 0.0, 45.0
    for _ in 1:100
        middle = (lower + upper) / 2
        yin_yang_owned(λ, middle) ? (lower = middle) : (upper = middle)
    end
    return (lower + upper) / 2
end

# The owned area fractions of the interior cells of a Yin-Yang panel, and the cell faces (degrees)
function panel_weights(grid)
    Nx, Ny, _ = size(grid)
    w = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)[1:Nx, 1:Ny, 1]
    return w, yin_yang_panel_face_coordinates(grid.connectivity.core_size, grid.connectivity.overlap)...
end

tilted_grid(N) = RotatedLatitudeLongitudeGrid(CPU(); size=(5N, 4N, 1), longitude=(-100, 60), latitude=(-50, 70),
                                              north_pole=(70, 55), z=(-1, 0))

cpu_grid = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0))
yin, yang = cpu_grid[1], cpu_grid[2]
Nx, Ny, _ = size(yin)
Hx, Hy, _ = halo_size(yin)
all_nodes = [(i, j) for j in 1-Hy:Ny+Hy for i in 1-Hx:Nx+Hx]

@testset "YinYangGrid construction and owned weights [$(typeof(arch)), $FT]" for arch in archs, FT in float_types
    grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))
    δ = grid.connectivity.overlap

    @test eltype(grid) == FT
    @test topology(grid) == (TwoSided{Connected, Connected}, TwoSided{Connected, Connected}, Bounded)
    @test δ == 3 # ⌈(2H + s) / 4⌉ for halo H = 3 and stencil width s = 4
    @test size(grid) == (48 + 2δ, 16 + 2δ, 2)
    @test typeof(grid[1]) == typeof(grid[2]) <: RotatedLatitudeLongitudeGrid
    @test CenterField(grid) isa YinYangField
    @test summary(grid) ==
        "54×22×2 YinYangGrid{$FT, Connected, Connected, Bounded} on $(summary(arch)) with 3×3×3 halo"
    @test_throws ArgumentError YinYangGrid(arch, FT; size=(40, 16, 2), z=(-1, 0))
    @test_throws ArgumentError YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0), overlap=5)

    w = owned_weights(grid)
    @test w isa YinYangField{Center, Center, Nothing}
    @test getregion(w, 1).data === getregion(w, 2).data
    weights = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)
    @test eltype(weights) == FT
    @test axes(weights) == (1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:1)
    @test all(iszero, weights[Nx+1:Nx+Hx, :, 1])

    # The panel area is analytic, and the owned cells of the two panels tile the sphere, to rounding error
    R = grid[1].radius
    rtol = FT == Float64 ? 1e-12 : 1e-4
    λᶠ, φᶠ = yin_yang_panel_face_coordinates((48, 16), δ)
    Az = on_architecture(CPU(), grid[1].Azᶜᶜᵃ)
    @test sum(Az[1:Nx, 1:Ny]) ≈ R^2 * deg2rad(λᶠ[end] - λᶠ[1]) * 2sind(φᶠ[end]) rtol=rtol
    @test 2 * sum(Float64(weights[i, j, 1]) * Az[i, j] for i in 1:Nx, j in 1:Ny) ≈ 4π * R^2 rtol=rtol

    wide_grid = with_halo((4, 4, 4), grid)
    @test (halo_size(wide_grid), size(wide_grid), wide_grid.connectivity.overlap) == ((4, 4, 4), size(grid), δ)

    # The panel coordinates are computed in the grid's precision, and agree with Float64 to that precision
    typed_yang = on_architecture(CPU(), grid[2])
    @test eltype(typed_yang.λᶜᶜᵃ) == FT
    @test maximum(abs, typed_yang.φᶜᶜᵃ[1:Nx, 1:Ny] .- yang.φᶜᶜᵃ[1:Nx, 1:Ny]) < (FT == Float64 ? 1e-12 : 1e-3)
end

@testset "Overlap validation" begin
    # At 2° the binding receivers sit exactly on donor centers and need δ ≥ 1 for 4×4 stencils
    @test_throws ArgumentError YinYangGrid(CPU(); size=(135, 45, 1), z=(-1, 0), overlap=0)
    @test YinYangGrid(CPU(); size=(135, 45, 1), z=(-1, 0), overlap=1) isa YinYangGrid
end

@testset "Kageyama-Sato involution" begin
    Random.seed!(0)
    points = [(360rand() - 180, asind(2rand() - 1)) for _ in 1:200]
    @test maximum(p -> angular_distance(p, yin_yang_transform(yin_yang_transform(p...)...)), points) < 1e-10

    # The panel coordinates of node (i, j) on Yang are the geographic coordinates of node (i, j) on Yin,
    # so the geographic coordinates of Yang's nodes are the involution of Yin's, halos included
    @test maximum(angular_distance(node_coordinates(i, j, yang, ℓx, ℓy),
                                   yin_yang_transform(node_coordinates(i, j, yin, ℓx, ℓy)...))
                  for (ℓx, ℓy) in yin_yang_locations, (i, j) in all_nodes) < 1e-10

    # The face-face node in the middle of Yang lies at (180°, 0), and Yang's grid north pole is at (90°, 0)
    i₀, j₀ = Nx ÷ 2 + 1, Ny ÷ 2 + 1
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

    # The Yin-to-Yang angle at a geographic point is the rotation angle of the Yang grid there
    @test maximum(abs(yin_yang_rotation_angle(node_coordinates(i, j, yang, ℓx, ℓy)...) - rotation_angle(i, j, yang, ℓx, ℓy))
                  for (ℓx, ℓy) in yin_yang_locations, (i, j) in all_nodes) < 1e-12

    # ψ rotates the Cartesian basis vectors of the geographic frame onto those of Yang's frame, which the
    # involution M maps to the geographic frame
    M = [-1 0 0; 0 0 1; 0 1 0]
    east(λ) = [-sind(λ), cosd(λ), 0]
    north(λ, φ) = [-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ)]
    Random.seed!(1234)
    for _ in 1:20
        λ, φ = 360rand() - 180, 170rand() - 85
        λ′, φ′ = yin_yang_transform(λ, φ)
        ψ = yin_yang_rotation_angle(λ, φ)
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

    # The owned fractions are symmetric, and those of cut cells match a midpoint rule on the ownership rule
    fine_grid = YinYangGrid(CPU(); size=(135, 45, 1), z=(-1, 0), overlap=1)
    for grid in (cpu_grid, fine_grid)
        w, λᶠ, φᶠ = panel_weights(grid)
        @test all(0 .≤ w .≤ 1)
        @test w == reverse(w, dims=1) == reverse(w, dims=2)
    end

    w, λᶠ, φᶠ = panel_weights(fine_grid)
    Az = fine_grid[1].Azᶜᶜᵃ ./ fine_grid[1].radius^2
    errors = map(findall(0 .< w .< 1)) do I
        i, j = Tuple(I)
        samples = range(λᶠ[i], λᶠ[i+1], length=401)
        φ★ = deg2rad.(bisected_dividing_curve_latitude.((samples[1:end-1] .+ samples[2:end]) ./ 2))
        φ₁, φ₂ = deg2rad(φᶠ[j]), deg2rad(φᶠ[j+1])
        owned_sines = sin.(clamp.(φ★, φ₁, φ₂)) .- sin.(clamp.(-φ★, φ₁, φ₂))
        owned_area = sum(owned_sines) * deg2rad(λᶠ[i+1] - λᶠ[i]) / 400
        abs(w[i, j] - owned_area / Az[i, j])
    end
    @test maximum(errors) < 1e-3
end

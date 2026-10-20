include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: with_halo
using Oceananigans.MultiRegion: YinYangConnectivity, YinYangPartition, YinYangField, default_yin_yang_overlap, yin_yang_panel_extents,
                                yin_yang_panel_face_coordinates, yin_yang_transform, yin_yang_rotation_angle, yin_yang_masks,
                                yin_yang_owned, yin_yang_cell_weights, owned_weights,
                                dividing_curve_latitude, dividing_curve_owned_longitudes
using Oceananigans.Operators: rotation_angle
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, rotate_coordinates

# Rotate the 2-vector (u, v) by the angle θ with the convention of `Operators.rotation_angle`
rotate_vector((u, v), θ) = (u * cos(θ) - v * sin(θ), u * sin(θ) + v * cos(θ))

# Angle of grid north at node (i, j) estimated by finite differences of the geographic
# coordinates of the two neighbors along j, with the convention of `Operators.rotation_angle`.
function finite_difference_rotation_angle(i, j, grid, ℓx, ℓy)
    ℓy_neighbor = ℓy isa Center ? Face() : Center()
    j₋, j₊ = ℓy isa Center ? (j, j + 1) : (j - 1, j)
    λ₋, φ₋ = node_coordinates(i, j₋, grid, ℓx, ℓy_neighbor)
    λ₊, φ₊ = node_coordinates(i, j₊, grid, ℓx, ℓy_neighbor)
    _, φ = node_coordinates(i, j, grid, ℓx, ℓy)
    east = wrap_longitude(λ₊ - λ₋) * cosd(φ)
    north = φ₊ - φ₋
    return atan(east, north)
end

# Maximum difference (in degrees) between the closed-form and finite-difference rotation angles
# of the `panels`, away from the geographic poles where the geographic basis is singular.
function maximum_rotation_angle_error(panels)
    error = 0.0
    for panel in panels, (ℓx, ℓy) in yin_yang_locations
        Nx, Ny, _ = size(panel)
        for j in 1:Ny, i in 1:Nx
            _, φ = node_coordinates(i, j, panel, ℓx, ℓy)
            abs(φ) > 80 && continue
            θ = rotation_angle(i, j, panel, ℓx, ℓy)
            θᶠᵈ = finite_difference_rotation_angle(i, j, panel, ℓx, ℓy)
            error = max(error, abs(wrap_longitude(rad2deg(θ - θᶠᵈ))))
        end
    end
    return error
end

maximum_rotation_angle_error(grid::YinYangGrid) = maximum_rotation_angle_error((grid[1], grid[2]))

# Collect the pairs of nodes ((i, j), (i′, j′)) at location (ℓx, ℓy) where a Yin node coincides with
# a Yang node, found by mapping every Yin node to fractional Yang indices.
function coincident_nodes(grid, ℓx, ℓy; tolerance = 1e-6)
    yin, yang = grid[1], grid[2]
    Nx, Ny, _ = size(yin)
    pairs = Tuple{NTuple{2, Int}, NTuple{2, Int}}[]
    for j in 1:Ny, i in 1:Nx
        λ, φ = node_coordinates(i, j, yin, ℓx, ℓy)
        i★, j★ = geographic_fractional_indices(λ, φ, grid, 2, ℓx, ℓy)
        i′ = round(Int, i★)
        j′ = round(Int, j★)
        inside = 1 ≤ i′ ≤ Nx && 1 ≤ j′ ≤ Ny
        on_node = abs(i★ - i′) < tolerance && abs(j★ - j′) < tolerance
        inside && on_node && push!(pairs, ((i, j), (i′, j′)))
    end
    return pairs
end

panel_area(panel) = sum(panel.Azᶜᶜᵃ[1:panel.Nx, 1:panel.Ny])

function analytic_panel_area(grid)
    connectivity = grid.connectivity
    λᵉ, φᵉ = yin_yang_panel_extents(connectivity.core_size, connectivity.overlap)
    R = grid[1].radius
    return R^2 * deg2rad(2λᵉ) * 2sind(φᵉ)
end

cell_areas(λᶠ, φᶠ) = [deg2rad(λᶠ[i+1] - λᶠ[i]) * (sind(φᶠ[j+1]) - sind(φᶠ[j])) for i in 1:length(λᶠ)-1, j in 1:length(φᶠ)-1]

# Latitude (degrees) of the dividing curve above longitude λ, by bisection of the ownership rule
function bisected_dividing_curve_latitude(λ)
    lower, upper = 0.0, 45.0
    for _ in 1:100
        middle = (lower + upper) / 2
        yin_yang_owned(λ, middle) ? (lower = middle) : (upper = middle)
    end
    return (lower + upper) / 2
end

@testset "YinYangGrid" begin
    for arch in archs
        @testset "Construction [$(typeof(arch)), $FT]" for FT in float_types
            grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))

            @test grid isa YinYangGrid
            @test grid.partition isa YinYangPartition
            @test grid.connectivity isa YinYangConnectivity
            @test eltype(grid) == FT
            @test length(grid) == 2
            @test topology(grid) == (FullyConnected, FullyConnected, Bounded)

            δ = default_yin_yang_overlap((3, 3, 3))
            @test δ == 3
            @test grid.connectivity.overlap == δ
            @test size(grid) == (48 + 2δ, 16 + 2δ, 2)
            @test all(yin_yang_panel_extents(grid.connectivity.core_size, δ) .≈ (135 + δ * 90 / 16, 45 + δ * 90 / 16))

            yin, yang = grid[1], grid[2]
            @test yin isa RotatedLatitudeLongitudeGrid
            @test typeof(yin) == typeof(yang)
            @test eltype(yin) == FT

            @test summary(grid) == "54×22×2 YinYangGrid{$FT, FullyConnected, FullyConnected, Bounded} on $(summary(arch)) with 3×3×3 halo"
            @test startswith(sprint(show, grid), summary(grid))

            c = CenterField(grid)
            @test c isa YinYangField

            @test_throws ArgumentError YinYangGrid(arch, FT; size=(40, 16, 2), z=(-1, 0))
            @test_throws ArgumentError YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0), overlap=5)

            new_grid = with_halo((4, 4, 4), grid)
            @test new_grid isa YinYangGrid
            @test halo_size(new_grid) == (4, 4, 4)
            @test size(new_grid) == size(grid)
            @test new_grid.connectivity.overlap == grid.connectivity.overlap
        end

        @testset "Owned weights [$(typeof(arch)), $FT]" for FT in float_types
            grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))
            Nx, Ny, _ = size(grid)
            Hx, Hy, _ = halo_size(grid)

            @test summary(grid.connectivity) == "YinYangConnectivity with 48×16 core, 3-cell overlap and 4×4 interpolation stencils"
            @test occursin("on every side, 4×4 interpolation stencils", sprint(show, grid))

            w = owned_weights(grid)
            @test w isa YinYangField
            @test location(w) == (Center, Center, Nothing)
            @test getregion(w, 1).data === getregion(w, 2).data

            weights = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)
            @test eltype(weights) == FT
            @test axes(weights) == (1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:1)
            @test all(iszero, weights[Nx+1:Nx+Hx, :, 1])

            # The owned cells of the two panels tile the sphere, to rounding error in the cell areas
            R = grid[1].radius
            Az = on_architecture(CPU(), grid[1].Azᶜᶜᵃ)
            total_area = 2 * sum(Float64(weights[i, j, 1]) * Az[i, j] for i in 1:Nx, j in 1:Ny)
            @test total_area ≈ 4π * R^2 rtol = FT == Float64 ? 1e-13 : 1e-4

            new_grid = with_halo((4, 4, 4), grid)
            @test axes(yin_yang_masks(new_grid).cell_weights) == (-3:Nx+4, -3:Ny+4, 1:1)
        end

        @testset "Overlap validation [$(typeof(arch))]" begin
            # At 2° the binding receivers sit exactly on donor centers and need δ ≥ 1 for 4×4 stencils
            @test_throws ArgumentError YinYangGrid(arch; size=(135, 45, 1), z=(-1, 0), overlap=0)
            @test YinYangGrid(arch; size=(135, 45, 1), z=(-1, 0), overlap=1) isa YinYangGrid

            @test default_yin_yang_overlap((5, 5, 5)) == 4
        end

        cpu_grid = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0))
        yin, yang = cpu_grid[1], cpu_grid[2]
        Nx, Ny, _ = size(yin)
        Hx, Hy, _ = halo_size(yin)

        @testset "Kageyama-Sato involution" begin
            # The map is its own inverse
            Random.seed!(0)
            error = 0.0
            for _ in 1:1000
                λ = 360rand() - 180
                φ = asind(2rand() - 1)
                λ″, φ″ = yin_yang_transform(yin_yang_transform(λ, φ)...)
                error = max(error, abs(wrap_longitude(λ″ - λ)) * cosd(φ), abs(φ″ - φ))
            end
            @test error < 1e-10
            @test all(yin_yang_transform(0, 0) .≈ (180, 0))

            # The panel coordinates of node (i, j) on Yang are the geographic coordinates of node (i, j)
            # on Yin, so the geographic coordinates of Yang's nodes are the involution of Yin's,
            # at every location, halos included.
            error = 0.0
            for (ℓx, ℓy) in yin_yang_locations, j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx
                λ′, φ′ = yin_yang_transform(node_coordinates(i, j, yin, ℓx, ℓy)...)
                λ, φ = node_coordinates(i, j, yang, ℓx, ℓy)
                error = max(error, abs(wrap_longitude(λ - λ′)) * cosd(φ), abs(φ - φ′))
            end
            @test error < 1e-10

            # The Yang panel point (λ′, φ′) = (0, 0) is the face-face node in the middle of the panel,
            # which lies at the geographic point (180°, 0). Yang's grid north pole is at (90°, 0).
            i₀, j₀ = Nx ÷ 2 + 1, Ny ÷ 2 + 1
            @test node_coordinates(i₀, j₀, yin, Face(), Face()) == (0, 0)
            λ, φ = node_coordinates(i₀, j₀, yang, Face(), Face())
            @test abs(wrap_longitude(λ - 180)) < 1e-10
            @test abs(φ) < 1e-10
            @test all(yang.conformal_mapping.north_pole .== (90, 0))
            @test all(rotate_coordinates(0, 90, 90, 0) .≈ (90, 0))
        end

        @testset "Coincident nodes in the overlap" begin
            f(λ, φ, z) = cosd(φ) * cosd(λ - 30) + sind(φ)^3 + z

            for (ℓx, ℓy) in yin_yang_locations
                pairs = coincident_nodes(cpu_grid, ℓx, ℓy)
                (ℓx, ℓy) == (Face(), Face()) && @test length(pairs) > 0

                distances = map(pairs) do ((i, j), (i′, j′))
                    λ, φ = node_coordinates(i, j, yin, ℓx, ℓy)
                    λ′, φ′ = node_coordinates(i′, j′, yang, ℓx, ℓy)
                    max(abs(wrap_longitude(λ - λ′)) * cosd(φ), abs(φ - φ′))
                end
                @test maximum(distances; init=0.0) < 1e-8

                # set! evaluates a function of geographic coordinates consistently on both panels
                field = Field{typeof(ℓx), typeof(ℓy), Center}(cpu_grid)
                set!(field, f)
                yin_field, yang_field = getregion(field, 1), getregion(field, 2)

                for ((i, j), (i′, j′)) in pairs, k in 1:2
                    @test yin_field[i, j, k] ≈ yang_field[i′, j′, k]
                end
            end
        end

        @testset "Panel areas" begin
            for panel in (yin, yang)
                @test panel_area(panel) ≈ analytic_panel_area(cpu_grid) rtol=1e-12
            end

            # With no overlap beyond the core, each panel covers 270° × 90°: an area of 3π/√2 R²,
            # so that the two panels cover 6π/√2 R² = 4πR² × (1 + 6.07%). No grid can be built
            # without overlap, since the faces on the panel edges then have no donor stencil, so
            # the core panels are built directly.
            @test_throws ArgumentError YinYangGrid(CPU(); size=(48, 16, 1), z=(-1, 0), overlap=0, stencil_width=2)
            R = yin.radius
            core_panels = (RotatedLatitudeLongitudeGrid(CPU(); size=(48, 16, 1), z=(-1, 0), radius=R, north_pole=(0, 90),
                                                        latitude=(-45, 45), longitude=(-135, 135)),
                           RotatedLatitudeLongitudeGrid(CPU(); size=(48, 16, 1), z=(-1, 0), radius=R, north_pole=(90, 0),
                                                        latitude=(-45, 45), longitude=(-45, 225)))
            for panel in core_panels
                @test panel_area(panel) ≈ 3π / √2 * R^2 rtol=1e-12
            end

            overlap_fraction = (panel_area(core_panels[1]) + panel_area(core_panels[2])) / (4π * R^2) - 1
            @test overlap_fraction ≈ 3 / (2√2) - 1 rtol=1e-10
            @test round(100 * overlap_fraction, digits=2) == 6.07
        end

        @testset "Rotation angles" begin
            # Closed-form angles converge to finite differences of the coordinates at second order
            coarse_error = maximum_rotation_angle_error(cpu_grid)
            fine_error = maximum_rotation_angle_error(YinYangGrid(CPU(); size=(96, 32, 1), z=(-1, 0)))
            @test coarse_error < 2
            @test fine_error < coarse_error / 3

            # Yin's panel frame is the geographic frame
            @test maximum(abs(rotation_angle(i, j, yin, ℓx, ℓy)) for (ℓx, ℓy) in yin_yang_locations, i in 1:Nx, j in 1:Ny) < 1e-12

            # The closed-form angle agrees with the finite-difference angle of `Operators.rotation_angle`
            # at cell centers, to the accuracy of the latter.
            error = 0.0
            for j in 1:Ny, i in 1:Nx
                _, φ = node_coordinates(i, j, yang, Center(), Center())
                abs(φ) > 80 && continue
                error = max(error, abs(rotation_angle(i, j, yang) - rotation_angle(i, j, yang, Center(), Center())))
            end
            @test error < deg2rad(1)

            # The Yin-to-Yang angle at a geographic point is the rotation angle of the Yang grid there
            error = 0.0
            for (ℓx, ℓy) in yin_yang_locations, j in 1-Hy:Ny+Hy, i in 1-Hx:Nx+Hx
                λ, φ = node_coordinates(i, j, yang, ℓx, ℓy)
                error = max(error, abs(yin_yang_rotation_angle(λ, φ) - rotation_angle(i, j, yang, ℓx, ℓy)))
            end
            @test error < 1e-12
        end

        @testset "Vector transforms between panels" begin
            Random.seed!(1234)

            for _ in 1:100
                λ = 360 * rand() - 180
                φ = 170 * rand() - 85
                ψ = yin_yang_rotation_angle(λ, φ)

                # R(ψ) R(ψ)ᵀ = I
                R = [cos(ψ) -sin(ψ); sin(ψ) cos(ψ)]
                @test R * transpose(R) ≈ I

                # The angle at the same point in the other panel's frame is -ψ,
                # so Yin → Yang → Yin is the identity
                λ′, φ′ = yin_yang_transform(λ, φ)
                ψ′ = yin_yang_rotation_angle(λ′, φ′)
                @test abs(sin(ψ + ψ′)) < 1e-10
                @test cos(ψ + ψ′) ≈ 1

                u, v = randn(), randn()
                u′, v′ = rotate_vector((u, v), ψ)
                u″, v″ = rotate_vector((u′, v′), ψ′)
                @test u″ ≈ u
                @test v″ ≈ v

                # Compare against the Cartesian basis vectors of both frames,
                # mapping Yang's basis to the geographic frame with M.
                M = [-1 0 0; 0 0 1; 0 1 0]
                east(λ) = [-sind(λ), cosd(λ), 0]
                north(λ, φ) = [-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ)]
                e′ = M * east(λ′)
                n′ = M * north(λ′, φ′)
                @test cos(ψ) ≈ east(λ) ⋅ e′
                @test sin(ψ) ≈ east(λ) ⋅ n′
            end
        end

        @testset "Panel coordinates in the grid's precision [$FT]" for FT in float_types
            grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))
            yang = on_architecture(CPU(), grid[2])
            @test eltype(yang.λᶜᶜᵃ) == FT
            @test rotate_coordinates(FT(10), FT(20), 90, 0) isa NTuple{2, FT}

            # The Float32 coordinates agree with the Float64 coordinates to Float32 precision
            yang64 = on_architecture(CPU(), YinYangGrid(arch, Float64; size=(48, 16, 2), z=(-1, 0))[2])
            tolerance = FT == Float64 ? 1e-12 : 1e-3
            @test maximum(abs, yang.φᶜᶜᵃ[1:Nx, 1:Ny] .- yang64.φᶜᶜᵃ[1:Nx, 1:Ny]) < tolerance
        end
    end
end

@testset "RotatedLatitudeLongitudeGrid support for Yin-Yang" begin
    for arch in archs
        tilted_grid = RotatedLatitudeLongitudeGrid(arch; size = (20, 16, 2),
                                                   longitude = (-100, 60),
                                                   latitude = (-50, 70),
                                                   north_pole = (70, 55),
                                                   z = (-1, 0))

        @testset "Closed-form rotation angle on a tilted RotatedLatitudeLongitudeGrid" begin
            # The geographic south pole is near the grid's southern edge; `maximum_rotation_angle_error`
            # excludes its neighborhood, where the geographic basis is singular.
            coarse_error = maximum_rotation_angle_error((on_architecture(CPU(), tilted_grid),))

            fine_grid = RotatedLatitudeLongitudeGrid(CPU(); size = (80, 64, 1),
                                                     longitude = (-100, 60),
                                                     latitude = (-50, 70),
                                                     north_pole = (70, 55),
                                                     z = (-1, 0))

            fine_error = maximum_rotation_angle_error((fine_grid,))
            @test fine_error < 0.5
            @test fine_error < coarse_error / 10

            # An unrotated frame has no rotation angle
            @test abs(rotated_basis_angle(10, 20, 0, 90)) < 1e-15
        end
    end
end

@testset "Yin-Yang ownership and owned area fractions" begin
    Random.seed!(1)
    exactly_one_owner = true
    owned_inside_core = true
    for _ in 1:10000
        λ = 360rand() - 180
        φ = asind(2rand() - 1)
        λ″, φ″ = yin_yang_transform(λ, φ)
        owned_by_yin = yin_yang_owned(λ, φ)
        owned_by_yang = yin_yang_owned(λ″, φ″)
        exactly_one_owner &= owned_by_yin + owned_by_yang == 1
        owned_inside_core &= owned_by_yin ? abs(λ) ≤ 135 && abs(φ) ≤ 45 : abs(λ″) ≤ 135 && abs(φ″) ≤ 45
    end
    @test exactly_one_owner
    @test owned_inside_core

    # The closed-form dividing curve and the owned longitudes of each row agree with the ownership rule
    λs = range(0, 134.99, length=200)
    @test rad2deg.(dividing_curve_latitude.(deg2rad.(λs))) ≈ bisected_dividing_curve_latitude.(λs) rtol=1e-12

    disagreements = 0
    for φ in range(0, 44.99, length=50), λ in range(0, 135, length=100)
        _, b₁, a₂, b₂ = rad2deg.(dividing_curve_owned_longitudes(deg2rad(φ)))
        min(abs(λ - b₁), abs(λ - a₂), abs(λ - b₂)) > 1e-6 || continue
        disagreements += yin_yang_owned(λ, φ) != (λ < b₁ || a₂ < λ < b₂)
    end
    @test disagreements == 0

    for (Nφ, overlap) in ((90, 0), (45, 2), (9, 0))
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
    errors = map(findall(0 .< w .< 1)) do I
        i, j = Tuple(I)
        samples = range(λᶠ[i], λᶠ[i+1], length=401)
        λs = (samples[1:end-1] .+ samples[2:end]) ./ 2
        φ★ = deg2rad.(bisected_dividing_curve_latitude.(λs))
        φ₁, φ₂ = deg2rad(φᶠ[j]), deg2rad(φᶠ[j+1])
        owned_area = sum(sin.(clamp.(φ★, φ₁, φ₂)) .- sin.(clamp.(.-φ★, φ₁, φ₂))) * deg2rad(λᶠ[i+1] - λᶠ[i]) / 400
        abs(w[i, j] - owned_area / cell_areas(λᶠ, φᶠ)[i, j])
    end
    @test maximum(errors) < 1e-3
end

include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Coriolis: f̃ᶠᶜᵃ, f̃ʸᶠᶜᵃ, x_f̃_cross_U, y_f̃_cross_U, z_f̃_cross_U,
                            HydrostaticFormulation, NonhydrostaticFormulation,
                            TriadScheme, ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving, DualGridScheme
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: extrinsic_vector, volume
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

# We set the grid-frame components of a smooth geographic velocity field on a rotated grid and compare the
# discrete Coriolis acceleration with the exact 2Ω × U, with a `LatitudeLongitudeGrid` as the reference for the error.
geographic_velocity(λ, φ) = (20 * cosd(φ) + 5 * sind(2λ) * cosd(φ), 10 * sind(λ) * cosd(φ)^2, cosd(λ) * cosd(φ))

# The exact Coriolis acceleration (east, north, up), or under the traditional approximation only f ẑ × U
function exact_coriolis(λ, φ, Ω, traditional)
    u, v, w = geographic_velocity(λ, φ)
    f = 2Ω * sind(φ)
    f̃ = traditional ? zero(f) : 2Ω * cosd(φ)
    return (f̃ * w - f * v, f * u, - f̃ * u)
end

# Cartesian unit vectors pointing east, north, and up at (λ, φ), and the Cartesian vector with those components
east_vector(λ, φ)  = [-sind(λ), cosd(λ), 0]
north_vector(λ, φ) = [-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ)]
up_vector(λ, φ)    = [cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ)]
cartesian_vector(λ, φ, (qᵉ, qⁿ, qᵘ)) = qᵉ * east_vector(λ, φ) + qⁿ * north_vector(λ, φ) + qᵘ * up_vector(λ, φ)

# The rotation from the rotated frame of a grid with `north_pole = (λ₀, φ₀)` to the geographic frame,
# as used to build a `RotatedLatitudeLongitudeGrid`; `(0, 90)` is the identity.
function frame_rotation(λ₀, φ₀)
    θ = 90 - φ₀
    Ry = [cosd(θ) 0 sind(θ); 0 1 0; -sind(θ) 0 cosd(θ)]
    Rz = [cosd(λ₀) -sind(λ₀) 0; sind(λ₀) cosd(λ₀) 0; 0 0 1]
    return Rz * Ry
end

# The rotated-frame coordinates of the geographic point (λ, φ)
function rotated_coordinates(λ, φ, R)
    x, y, z = R' * up_vector(λ, φ)
    return atand(y, x), asind(z)
end

function test_grids(arch, N; north_pole)
    kw = (size = (N, N, 2), halo = (3, 3, 3), z = (0, 1), longitude = (40, 80), latitude = (-20, 20))
    rotated_grid = RotatedLatitudeLongitudeGrid(arch; north_pole, topology = (Bounded, Bounded, Bounded), kw...)
    return rotated_grid, LatitudeLongitudeGrid(arch; kw...)
end

# The component along the grid's x̂ (`axis = east_vector`), ŷ (`north_vector`) or ẑ (`up_vector`) of the geographic
# vector field `vector(λ, φ) = (east, north, up)` at the (ℓx, ℓy) point `i, j` of a grid with frame rotation `R`
function grid_frame_component(i, j, ℓx, ℓy, axis, grid, R, vector)
    λ = λnode(i, j, 1, grid, ℓx, ℓy, Center())
    φ = φnode(i, j, 1, grid, ℓx, ℓy, Center())
    λ′, φ′ = rotated_coordinates(λ, φ, R)
    return cartesian_vector(λ, φ, vector(λ, φ)) ⋅ (R * axis(λ′, φ′))
end

# Set `q` to `value(i, j, k)` everywhere, halos included
function set_with_halos!(q, value)
    for I in CartesianIndices(q.data)
        q.data[I] = value(Tuple(I)...)
    end
    return q
end

# The grid-frame components of the geographic velocity
function geographic_velocity_fields(grid, R)
    u(i, j, k) = grid_frame_component(i, j, Face(), Center(), east_vector, grid, R, geographic_velocity)
    v(i, j, k) = grid_frame_component(i, j, Center(), Face(), north_vector, grid, R, geographic_velocity)
    w(i, j, k) = grid_frame_component(i, j, Center(), Center(), up_vector, grid, R, geographic_velocity)
    return set_with_halos!(XFaceField(grid), u), set_with_halos!(YFaceField(grid), v), set_with_halos!(ZFaceField(grid), w)
end

function coriolis_accelerations(grid, coriolis, U)
    ax = Field(KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, grid, coriolis, U))
    ay = Field(KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, grid, coriolis, U))
    az = Field(KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, grid, coriolis, U))
    return compute!(ax), compute!(ay), compute!(az)
end

# The maximum errors of the discrete Coriolis acceleration, relative to the largest exact acceleration:
# in the grid-frame components at the velocity points, and in the geographic components at cell centers
function coriolis_errors(grid, R, coriolis; traditional = false)
    ax, ay, az = coriolis_accelerations(grid, coriolis, geographic_velocity_fields(grid, R))
    exact(λ, φ) = exact_coriolis(λ, φ, coriolis.rotation_rate, traditional)

    grid_frame_error = geographic_error = scale = 0.0
    for k in 1:size(grid, 3), j in 1:size(grid, 2), i in 1:size(grid, 1)
        grid_frame_error = max(grid_frame_error,
                               abs(ax[i, j, k] - grid_frame_component(i, j, Face(), Center(), east_vector, grid, R, exact)),
                               abs(ay[i, j, k] - grid_frame_component(i, j, Center(), Face(), north_vector, grid, R, exact)),
                               abs(az[i, j, k] - grid_frame_component(i, j, Center(), Center(), up_vector, grid, R, exact)))

        λ = λnode(i, j, k, grid, Center(), Center(), Center())
        φ = φnode(i, j, k, grid, Center(), Center(), Center())
        aᵉ, aⁿ, _ = exact(λ, φ)
        a₁, a₂ = extrinsic_vector(i, j, k, grid, (ax[i, j, k] + ax[i+1, j, k]) / 2, (ay[i, j, k] + ay[i, j+1, k]) / 2)
        geographic_error = max(geographic_error, abs(a₁ - aᵉ), abs(a₂ - aⁿ))
        scale = max(scale, abs(aᵉ), abs(aⁿ))
    end

    return grid_frame_error / scale, geographic_error / scale
end

@testset "SphericalCoriolis on rotated grids" begin
    # The comparison sets and reads fields one point at a time, so it runs on the CPU
    arch = CPU()
    Oceananigans.defaults.FloatType = Float64
    north_pole = (0, 0)
    R = frame_rotation(north_pole...)

    @testset "Rotated grid matches the latitude-longitude grid to discretization error" begin
        # The active-weighted schemes are left out: their wet-point correction counts the wall faces of this
        # bounded grid as inactive, which is only consistent with a velocity that vanishes there.
        for (scheme, formulation) in ((EnstrophyConserving(), NonhydrostaticFormulation()),
                                      (TriadScheme(), NonhydrostaticFormulation()),
                                      (EnergyConserving(), HydrostaticFormulation()))
            @testset "$(summary(scheme)) $(summary(formulation))" begin
                coriolis = SphericalCoriolis(; scheme, formulation)
                traditional = formulation isa HydrostaticFormulation
                grids = map(N -> test_grids(arch, N; north_pole), (8, 16))
                rotated_errors = map(((grid, _),) -> coriolis_errors(grid, R, coriolis; traditional), grids)
                reference_errors = map(((_, grid),) -> coriolis_errors(grid, frame_rotation(0, 90), coriolis; traditional), grids)

                for (rotated, reference) in zip(rotated_errors, reference_errors)
                    # Discretization errors are small, and on the rotated grid comparable to the reference
                    @test all(<(1e-2), rotated)
                    @test all(<(1e-2), reference)
                    @test all(rotated .< 4 .* reference)
                end

                # Second-order convergence
                coarse, fine = rotated_errors
                @test all(coarse ./ fine .> 3)
            end
        end
    end

    @testset "The nonhydrostatic formulation reduces to the hydrostatic one without the nontraditional terms" begin
        # The nontraditional terms multiply w in the horizontal components and u, v in the vertical one,
        # so with w ≡ 0 every scheme's horizontal acceleration is its traditional stencil exactly, and with
        # u ≡ v ≡ 0 the vertical acceleration vanishes. The hydrostatic formulation has no vertical component.
        grid, _ = test_grids(arch, 8; north_pole)
        for scheme in (EnstrophyConserving(), EnergyConserving(), TriadScheme(),
                       ActiveWeightedEnstrophyConserving(), ActiveWeightedEnergyConserving())
            nonhydrostatic = SphericalCoriolis(; scheme)
            hydrostatic = HydrostaticSphericalCoriolis(; scheme)

            u, v, w = geographic_velocity_fields(grid, R)
            _, _, az_hydrostatic = coriolis_accelerations(grid, hydrostatic, (u, v, w))
            @test all(iszero, interior(az_hydrostatic))

            fill!(w, 0)
            ax, ay, _ = coriolis_accelerations(grid, nonhydrostatic, (u, v, w))
            ax_hydrostatic, ay_hydrostatic, _ = coriolis_accelerations(grid, hydrostatic, (u, v, w))
            @test interior(ax) == interior(ax_hydrostatic)
            @test interior(ay) == interior(ay_hydrostatic)

            u, v, w = geographic_velocity_fields(grid, R)
            fill!(u, 0)
            fill!(v, 0)
            _, _, az = coriolis_accelerations(grid, nonhydrostatic, (u, v, w))
            @test all(iszero, interior(az))
        end
    end

    @testset "The nontraditional terms do no net work" begin
        # With the cell volumes V = Az Δz, Σ u V (f̃ʸ w)ᵤ − Σ w V (f̃ʸ u)_w = 0 and likewise for the v–w pair,
        # provided the normal velocities vanish on the boundaries so that every product is counted on both
        # sides. The wall and top/bottom values of the analytic field are therefore set to zero.
        grid, _ = test_grids(arch, 12; north_pole)
        Nx, Ny, Nz = size(grid)
        coriolis = SphericalCoriolis()
        u, v, w = U = geographic_velocity_fields(grid, R)
        interior(u)[[1, Nx+1], :, :] .= 0
        interior(v)[:, [1, Ny+1], :] .= 0
        interior(w)[:, :, [1, Nz+1]] .= 0

        work = scale = 0.0
        for (q, ℓ, f̃_cross_U) in ((u, (Face(), Center(), Center()), x_f̃_cross_U),
                                  (v, (Center(), Face(), Center()), y_f̃_cross_U),
                                  (w, (Center(), Center(), Face()), z_f̃_cross_U))
            for I in CartesianIndices(interior(q))
                i, j, k = Tuple(I)
                δ = q[i, j, k] * volume(i, j, k, grid, ℓ...) * f̃_cross_U(i, j, k, grid, coriolis, U)
                work += δ
                scale += abs(δ)
            end
        end

        @test scale > 0
        @test abs(work) < 1e-12 * scale
    end

    @testset "Unsupported schemes are rejected rather than dropping the nontraditional terms" begin
        grid, _ = test_grids(arch, 8; north_pole)
        @test_throws ArgumentError SphericalCoriolis(scheme = DualGridScheme(grid))
        @test HydrostaticSphericalCoriolis(scheme = DualGridScheme(grid)) isa SphericalCoriolis
    end

    @testset "Unrotated grids" begin
        # A rotated grid whose pole is the geographic pole is a latitude-longitude grid
        N = 8
        unrotated_grid, latitude_longitude_grid = test_grids(arch, N; north_pole = (0, 90))
        coriolis = SphericalCoriolis()
        accelerations = map((latitude_longitude_grid, unrotated_grid)) do grid
            coriolis_accelerations(grid, coriolis, geographic_velocity_fields(grid, frame_rotation(0, 90)))
        end

        for (a_latitude_longitude, a_unrotated) in zip(accelerations...)
            @test interior(a_unrotated) ≈ interior(a_latitude_longitude) rtol=1e-10
        end

        # On the latitude-longitude grid the horizontal rotation vector is f̃ along y
        grid = latitude_longitude_grid
        @test all(f̃ʸᶠᶜᵃ(i, j, 1, grid, coriolis) === f̃ᶠᶜᵃ(i, j, 1, grid, coriolis) for j in 1:N, i in 1:N+1)
    end
end
